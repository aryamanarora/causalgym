import torch
import os
import random
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from plotnine import ggplot, geom_point, aes, facet_grid, geom_line, ggtitle, geom_tile, theme, element_text, facet_wrap, geom_text
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse, scale_fill_gradient2, scale_fill_gradient
from utils import MODELS, WEIGHTS, get_last_token
from data import make_data
from eval import calculate_loss, eval, eval_sentence
import plot
import datetime
import json

# make a timestamp int
NOW = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.basic_utils import format_token, sm, count_parameters
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
    intervention_class = VanillaIntervention
    if intervention_obj is None:
        if num_dims == -1:
            intervention_class = BoundlessRotatedSpaceIntervention
        elif num_dims > 0:
            intervention_class = LowRankRotatedSpaceIntervention
    else:
        intervention_class = type(intervention_obj)
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
        alignable_interventions_type=intervention_class,
        alignable_interventions=[intervention_obj]
    )
    return alignable_config

def experiment(
    model: str,
    dataset: str,
    steps: int,
    num_dims: int,
    warmup: bool,
    eval_steps: int,
    grad_steps: int,
    batch_size: int,
    num_tokens: int,
    position: str,
    intervention_site: str,
    do_swap: bool=True,
    test_sentence: bool=False
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

    # sentence for evals
    sentence = "<|endoftext|>Jane is a terrible, evil person. John and Bob are very nice friends of ours. We will always talk to Jane."

    # make das subdir
    if not os.path.exists("figs/das"):
        os.makedirs("figs/das")
    if not os.path.exists("figs/das/steps"):
        os.makedirs("figs/das/steps")
    
    # clear files from figs/das/steps
    for file in os.listdir("figs/das/steps"):
        os.remove(os.path.join("figs/das/steps", file))

    # intervene on each layer
    data, scores = [], []
    stats = {}
    layer_objs = {}
    
    # entering train loops
    max_loop = 1
    pos_i = 0
    while pos_i < (max_loop if position == "each" else 1):

        # train and eval sets
        trainset, labels = make_data(tokenizer, dataset, batch_size, steps, num_tokens, device, position=position if position != "each" else pos_i, seed=42)
        evalset, _ = make_data(tokenizer, dataset, batch_size, 20, num_tokens, device, position=position if position != "each" else pos_i, seed=420)
        max_loop = len(trainset[0].pair[0].input_ids[0])

        # tokens to log
        tokens = tokenizer.encode("".join(labels))

        for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
            print(pos_i, layer_i)

            # set up alignable model
            alignable_config = intervention_config(
                type(gpt), intervention_site, layer_i, num_dims
            )
            alignable = AlignableModel(alignable_config, gpt)
            alignable.set_device(device)
            alignable.disable_model_gradients()

            # optimizer
            if num_dims != 0:
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
            else:
                total_steps = 1
                warm_up_steps = 0.1 * total_steps if warmup else 0

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
                (pair, src_label, base_label, pos_interv) = trainset[step]
                for i in range(2):

                    # inference
                    _, counterfactual_outputs = alignable(
                        pair[0],
                        [pair[1]],
                        {"sources->base": (pos_interv, pos_interv)},
                    )

                    # get last token logits
                    logits = get_last_token(counterfactual_outputs.logits, pair[0].attention_mask)

                    # loss and backprop
                    loss = calculate_loss(logits, src_label, step, alignable, warm_up_steps)
                    total_loss += loss
                    
                    # swap
                    if do_swap:
                        pair[0], pair[1] = pair[1], pair[0]
                        src_label, base_label = base_label, src_label
                    else:
                        break

                # gradient accumulation
                if total_step % grad_steps == 0 and num_dims != 0:
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
                if (step % eval_steps == 0 or step == total_steps - 1):
                    more_data, more_stats = eval(
                        alignable, tokenizer, evalset, step=step,
                        layer_i=layer_i, num_dims=num_dims, tokens=tokens
                    )
                    data.extend(more_data)
                    stats.update(more_stats)
                    iterator.set_postfix(stats)
                    layer_objs[layer_i] = alignable
                    prefix = str(step).zfill(4)
                    
                    if test_sentence:
                        scores, _ = eval_sentence(
                            tokenizer=tokenizer,
                            df=pd.DataFrame(data),
                            scores=scores,
                            layer_objs=layer_objs,
                            tokens=tokens,
                            evalset=evalset,
                            sentence=sentence,
                            dataset=dataset,
                            model=model,
                            prefix=f"steps/step_{prefix}",
                            step=step,
                            plots=True if (layer_i == gpt.config.num_hidden_layers - 1) else False,
                            layer=layer_i
                        )

                total_step += 1
            layer_objs[layer_i] = alignable
        pos_i += 1

    # make data dump
    short_dataset_name = dataset.split('/')[-1]
    short_model_name = model.split('/')[-1]
    filedump = {
        "metadata": {
            "model": model,
            "dataset": dataset,
            "steps": steps,
            "num_dims": num_dims,
            "warmup": warmup,
            "eval_steps": eval_steps,
            "grad_steps": grad_steps,
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "position": position,
            "intervention_site": intervention_site,
            "do_swap": do_swap,
            "scores": scores,
            "test_sentence": test_sentence,
        },
        "data": data
    }
    with open(f"logs/das/{short_model_name}__{short_dataset_name}__{NOW}.json", "w") as f:
        json.dump(filedump, f)

    # print plots
    df = pd.DataFrame(data)
    plot.plot_bounds(df, f"{short_dataset_name}, {short_model_name}: intervention boundary")
    plot.plot_label_loss(df, f"{short_dataset_name}, {short_model_name}: per-label loss")
    plot.plot_label_prob(df, f"{short_dataset_name}, {short_model_name}: per-label probs")
    plot.plot_label_logit(df, f"{short_dataset_name}, {short_model_name}: per-label logits")

    # cosine sim of learned directions plot
    if num_dims == 1:
        plot.plot_das_cos_sim(layer_objs, f"{short_dataset_name}, {short_model_name}: cosine similarity of learned directions")
    if position == "each":
        sentence = evalset[0].pair[0].input_ids[0]
        sentence = [format_token(tokenizer, tok) for tok in sentence]
        print(sentence)
        plot.plot_pos_iia(df, f"{short_dataset_name}, {short_model_name}: position iia", sentence=sentence)

    # plot
    if test_sentence:
        eval_sentence(
            tokenizer=tokenizer,
            df=df,
            layer_objs=layer_objs,
            tokens=tokens,
            evalset=evalset,
            step=total_steps - 1,
            sentence=sentence,
            dataset=dataset,
            model=model,
            plots=True
        )

    # make gif of files in figs/das/steps
    os.system("convert -delay 100 -loop 0 figs/das/steps/*prob_per_pos.png figs/das/prob_steps.gif")
    # os.system("convert -delay 100 -loop 0 figs/das/steps/*val_per_pos.png figs/das/val_steps.gif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="gender_basic")
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--num-dims", type=int, default=-1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--grad-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=-1)
    parser.add_argument("--position", type=str, default="all")
    parser.add_argument("--intervention-site", type=str, default="block_output")
    args = parser.parse_args()
    print(vars(args))
    experiment(**vars(args))


if __name__ == "__main__":
    main()
