import torch
import os
import random
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from plotnine import ggplot, geom_point, aes, facet_grid, geom_line, ggtitle, geom_tile, theme, element_text, facet_wrap
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse, scale_fill_gradient2, scale_fill_gradient
from utils import MODELS, WEIGHTS, Sentence, get_options, make_sentence
from data import make_data
from eval import calculate_loss, eval, eval_sentence

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import format_token, sm, count_parameters
from models.configuration_alignable_model import (
    AlignableRepresentationConfig,
    AlignableConfig,
)
from models.alignable_base import AlignableModel
from interventions import (
    LowRankRotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    VanillaIntervention
)

def intervention_config(model_type, intervention_type, layer, num_dims, intervention_obj=None):
    if intervention_obj is None:
        intervention_obj = BoundlessRotatedSpaceIntervention if num_dims == -1 else LowRankRotatedSpaceIntervention
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
                "pos",  # intervention unit
                1,  # max number of unit
                alignable_low_rank_dimension=num_dims,  # low rank dimension
            ),
        ],
        alignable_interventions_type=intervention_obj
    )
    return alignable_config

def get_last_token(logits, attention_mask):
    last_token_indices = attention_mask.sum(1) - 1
    batch_indices = torch.arange(logits.size(0)).unsqueeze(1)
    return logits[batch_indices, last_token_indices.unsqueeze(1)].squeeze(1)

def experiment(
    model: str,
    dataset: str,
    steps: int,
    num_dims: int,
    warmup: bool,
    eval_steps: int,
    grad_steps: int,
    batch_size: int,
):
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)
    print(gpt.config.num_hidden_layers)

    # train and eval sets
    trainset, labels = make_data(tokenizer, dataset, batch_size, steps, device)
    evalset, _ = make_data(tokenizer, dataset, 1, 20, device)

    # tokens to log
    tokens = tokenizer.encode("".join(labels))

    # intervene on each layer
    data = []
    stats = {}
    layer_objs = {}

    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):

        # set up alignable model
        alignable_config = intervention_config(
            type(gpt), "block_output", layer_i, num_dims
        )
        alignable = AlignableModel(alignable_config, gpt)
        alignable.set_device(device)
        alignable.disable_model_gradients()

        # optimizer
        total_steps = steps
        warm_up_steps = 0.1 * total_steps if warmup else 0
        optimizer_params = []
        for k, v in alignable.interventions.items():
            try:
                optimizer_params.append({"params": v[0].rotate_layer.parameters()})
                optimizer_params.append({"params": v[0].intervention_boundaries, "lr": 1e-2})
            except:
                print("some trainable params not found")
                pass
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=warm_up_steps,
        #     num_training_steps=total_steps,
        # )
        print("model trainable parameters: ", count_parameters(alignable.model))
        print("intervention trainable parameters: ", alignable.count_parameters())

        # temperature for boundless
        total_step = 0
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, total_steps)
            .to(torch.bfloat16)
            .to(device)
        )
        alignable.set_temperature(temperature_schedule[total_step])

        # train
        iterator = tqdm(range(total_steps))
        total_loss = torch.tensor(0.0).to(device)

        for step in iterator:

            # make pair
            (pair, src_label, _, pos_i) = trainset[step]

            # inference
            _, counterfactual_outputs = alignable(
                pair[0],
                [pair[1]],
                {"sources->base": (pos_i, pos_i)},
            )

            # get last token logits
            logits = get_last_token(counterfactual_outputs.logits, pair[0].attention_mask)

            # loss and backprop
            loss = calculate_loss(logits, src_label, step, alignable, warm_up_steps)
            total_loss += loss

            # gradient accumulation
            if total_step % grad_steps == 0:
                # print stats
                stats["lr"] = scheduler.optimizer.param_groups[0]['lr']
                stats["loss"] = f"{total_loss.item():.3f}"
                if num_dims == -1:
                    for k, v in alignable.interventions.items():
                        stats["bound"] = f"{v[0].intervention_boundaries.sum() * v[0].embed_dim:.3f}"
                iterator.set_postfix(stats)

                if not (grad_steps > 1 and total_step == 0):
                    total_loss.backward()
                    total_loss = torch.tensor(0.0).to(device)
                    optimizer.step()
                    scheduler.step()
                    alignable.set_zero_grad()
                    alignable.set_temperature(temperature_schedule[total_step])

            # eval
            if step % eval_steps == 0 or step == total_steps - 1:
                more_data, more_stats = eval(
                    alignable, tokenizer, evalset, step=step,
                    layer_i=layer_i, num_dims=num_dims, tokens=tokens
                )
                eval_sentence(
                    alignable=alignable,
                    tokenizer=tokenizer,
                    df=df,
                    layer_objs=layer_objs,
                    tokens=tokens,
                    evalset=evalset,
                    sentences="<|endoftext|>Jane went home because she was beautiful. My buddy John is my girlfriend's brother and he wants to be a nurse."
                )
                data.extend(more_data)
                stats.update(more_stats)

            total_step += 1
        if layer_i not in layer_objs:
            layer_objs[layer_i] = (alignable, loss.item())
        elif loss.item() < layer_objs[layer_i][1]:
            layer_objs[layer_i] = (alignable, loss.item())

    # make das subdir
    if not os.path.exists("figs/das"):
        os.makedirs("figs/das")

    # print plots
    df = pd.DataFrame(data)
    
    plot = (
        ggplot(df, aes(x="step", y="bound", color="factor(layer)"))
        + geom_line()
        + ggtitle("intervention boundary")
    )
    plot.save("figs/das/bound.pdf")

    plot = (
        ggplot(df, aes(x="step", y="loss", color="factor(label)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label loss")
    )
    plot.save("figs/das/loss.pdf")

    plot = (
        ggplot(df, aes(x="step", y="prob", color="factor(label_token)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label probs")
    )
    plot.save("figs/das/prob.pdf")

    plot = (
        ggplot(df, aes(x="step", y="logit", color="factor(label_token)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label logits")
    )
    plot.save("figs/das/logit.pdf")
        
    # plot
    scores_df, test = eval_sentence(
        alignable=alignable,
        tokenizer=tokenizer,
        df=df,
        layer_objs=layer_objs,
        tokens=tokens,
        evalset=evalset,
        sentences="<|endoftext|>Jane went home because she was beautiful. My buddy John is my girlfriend's brother and he wants to be a nurse."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="gender_basic")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--num_dims", type=int, default=-1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    print(vars(args))
    experiment(**vars(args))


if __name__ == "__main__":
    main()
