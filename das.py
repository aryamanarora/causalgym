import torch
import os
import random
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from plotnine import ggplot, geom_point, aes, facet_grid, geom_line, ggtitle, geom_tile, theme, element_text
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse
from utils import MODELS, WEIGHTS, Sentence, get_options, make_sentence

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import format_token, sm, count_parameters
from models.configuration_alignable_model import (
    AlignableRepresentationConfig,
    AlignableConfig,
)
from models.alignable_base import AlignableModel
from models.interventions import (
    VanillaIntervention,
    RotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
)

names = {
    "he": ["John", "Bill", "Joseph", "Patrick", "Ken", "Geoff", "Simon", "Richard", "David", "Michael"],
    "she": ["Sarah", "Mary", "Elizabeth", "Jane"]
}

def rotated_space_intervention(num_dims):
    def func(args, proj_dim):
        intervention = RotatedSpaceIntervention(args)
        intervention.set_interchange_dim(num_dims)
        return intervention
    return func

def intervention_config(model_type, intervention_type, layer, num_dims, intervention_obj=None):
    if intervention_obj is None:
        intervention_obj = BoundlessRotatedSpaceIntervention if num_dims == -1 else rotated_space_intervention(num_dims)
    else:
        def func(args, proj_dim):
            return intervention_obj
        intervention_obj = func
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
                "pos",  # intervention unit
                1,  # max number of unit
            ),
        ],
        alignable_interventions_type=intervention_obj,
    )
    return alignable_config


def experiment(
    model: str="EleutherAI/pythia-70m",
    steps: int=1000,
    num_dims: int=-1,
    warmup: bool=False,
    eval_steps: int=20,
    grad_steps: int=4
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

    # filter names
    for key in names:
        names[key] = [name for name in names[key] if len(tokenizer(name)['input_ids']) == 1]
    print(names)

    def make_pair():
        genders = ["he", "she"]
        random.shuffle(genders)
        gender1 = genders[0]
        gender2 = genders[1]
        he = random.choice(names[gender1])
        she = random.choice(names[gender2])
        completions = [
            "walked",
            # "is tired", "is excited", "is ready", "went home", "walked", "is walking",
            # "ran", "is running", "works there", "joined the army", "plays soccer",
            # "likes playing games", "said no to me"
        ]
        completion = random.choice(completions)
        pair = (
            tokenizer(f"<|endoftext|>{he} {completion} because", return_tensors="pt").to(device),
            tokenizer(f"<|endoftext|>{she} {completion} because", return_tensors="pt").to(device),
        )
        return pair, " " + gender2, " " + gender1

    tokens = tokenizer.encode(" she he")  # token we want to maximize the probability of

    # make evalset
    evalset = []
    for i in range(20):
        evalset.append(make_pair())

    # intervene on each layer
    # only intervening on layer 0, pos 1, dim 1
    data = []
    layer_objs = {}

    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
    # for layer_i in [0]:

        # for pos_i in range(1, len(base.input_ids[0])):
        for pos_i in [1]:

            # set up alignable model
            alignable_config = intervention_config(
                type(gpt), "block_output", layer_i, num_dims
            )
            alignable = AlignableModel(alignable_config, gpt)
            alignable.set_device(device)
            alignable.disable_model_gradients()

            # optimizer
            t_total = steps
            warm_up_steps = 0.1 * t_total if warmup else 0
            optimizer_params = []
            for k, v in alignable.interventions.items():
                optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
                try:
                    optimizer_params += [
                        {"params": v[0].intervention_boundaries, "lr": 1e-2}
                    ]
                except:
                    pass

            print("model trainable parameters: ", count_parameters(alignable.model))
            print(
                "intervention trainable parameters: ", alignable.count_parameters()
            )

            optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
                num_training_steps=t_total,
            )

            total_step = 0
            temperature_start = 50.0
            temperature_end = 0.1
            temperature_schedule = (
                torch.linspace(temperature_start, temperature_end, t_total)
                .to(torch.bfloat16)
                .to(device)
            )
            alignable.set_temperature(temperature_schedule[total_step])

            def calculate_loss(logits, label, step):
                shift_logits = logits[..., :, :].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits[0, -1]
                shift_labels = torch.tensor(label)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                if step >= warm_up_steps:
                    try:
                        for k, v in alignable.interventions.items():
                            boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
                        loss += boundary_loss
                    except:
                        pass

                return loss

            iterator = tqdm(range(t_total))
            total_loss = torch.tensor(0.0).to(device)
            for step in iterator:
                # make pair
                pair, label, _ = make_pair()

                # inference
                if step == 0: print(pos_i)
                _, counterfactual_outputs = alignable(
                    pair[0],
                    [pair[1]],
                    {"sources->base": ([[[pos_i]]], [[[pos_i]]])},
                )

                # loss and backprop
                loss = calculate_loss(
                    counterfactual_outputs.logits, tokenizer.encode(label)[0], step
                )
                total_loss += loss

                # gradient accumulation
                if total_step % grad_steps == 0:
                    # print stats
                    stats = {"loss": f"{total_loss.item():.3f}"}
                    distrib = sm(counterfactual_outputs.logits)[0, -1]
                    for tok in tokens:
                        prob = distrib[tok].item()
                        stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
                    try:
                        for k, v in alignable.interventions.items():
                            stats[
                                "bound"
                            ] = f"{v[0].intervention_boundaries.sum() * v[0].embed_dim:.3f}"
                    except:
                        pass
                    iterator.set_postfix(stats)

                    if not (grad_steps > 1 and total_step == 0):
                        total_loss.backward()
                        total_loss = torch.tensor(0.0).to(device)
                        optimizer.step()
                        scheduler.step()
                        alignable.set_zero_grad()
                        alignable.set_temperature(temperature_schedule[total_step])

                # eval
                if step % eval_steps == 0:
                    with torch.no_grad():
                        boundary = num_dims
                        try:
                            for k, v in alignable.interventions.items():
                                boundary = (
                                    v[0].intervention_boundaries.sum() * v[0].embed_dim
                                ).item()
                        except:
                            pass

                        for pair, label, base_label in evalset:
                            _, counterfactual_outputs = alignable(
                                pair[0],
                                [pair[1]],
                                {"sources->base": ([[[pos_i]]], [[[pos_i]]])},
                            )
                            loss = calculate_loss(
                                counterfactual_outputs.logits,
                                tokenizer.encode(label)[0],
                                step
                            )
                            distrib = sm(counterfactual_outputs.logits)[0, -1]
                            for tok in tokens:
                                prob = distrib[tok].item()
                                stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
                                data.append(
                                    {
                                        "step": step,
                                        "src_label": label,
                                        "base_label": base_label,
                                        "label": label + " > " + base_label,
                                        "loss": loss.item(),
                                        "token": format_token(tokenizer, tok),
                                        "label_token": label + " > " + base_label + " " + format_token(tokenizer, tok),
                                        "prob": prob,
                                        "bound": boundary,
                                        "temperature": temperature_schedule[total_step],
                                        "layer": layer_i,
                                        "pos": pos_i,
                                    }
                                )

                total_step += 1
            if layer_i not in layer_objs:
                layer_objs[layer_i] = (alignable, loss.item(), pos_i)
            elif loss.item() < layer_objs[layer_i][1]:
                layer_objs[layer_i] = (alignable, loss.item(), pos_i)

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
        + facet_grid("~layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label loss")
    )
    plot.save("figs/das/loss.pdf")

    plot = (
        ggplot(df, aes(x="step", y="prob", color="factor(label_token)"))
        + facet_grid("~layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label probs")
    )
    plot.save("figs/das/prob.pdf")

    # test probe on a sentence
    with torch.no_grad():
        test = tokenizer(
            "<|endoftext|>My buddy John is my girlfriend's brother and he wants to be a nurse.",
            #  Spain is a beautiful, cute, and handsome country.
            return_tensors="pt",
        ).to(device)

        scores = []
        for layer_i in layer_objs:
            for (pair, label, base_label) in tqdm(evalset):
                base_logits = gpt(**pair[0]).logits[0, -1]
                # alignable_config = intervention_config(
                #     type(gpt), "block_output", layer_i, -1, layer_objs[layer_i]
                # )
                alignable = layer_objs[layer_i][0]
                pos_base = layer_objs[layer_i][2]
                alignable.set_device(device)
                _, counterfactual_outputs = alignable(
                    pair[0],
                    [pair[1]],
                    {"sources->base": ([[[pos_base]]], [[[pos_base]]])},
                )
                logits = counterfactual_outputs.logits[0, -1]
                token_ranges = {}
                for token in tokens:
                    token_ranges[token] = (
                        base_logits[token].item(),
                        logits[token].item(),
                    )
                # print(token_ranges)
                
                for pos_i in range(1, len(test.input_ids[0])):
                    _, counterfactual_outputs = alignable(
                        pair[0], [test], {"sources->base": ([[[pos_i]]], [[[pos_base]]])}
                    )
                    logits = counterfactual_outputs.logits[0, -1]
                    partial_probs = logits.softmax(dim=-1)[tokens]
                    partial_probs = partial_probs / partial_probs.sum()
                    for i, token in enumerate(tokens):
                        score = 0.0
                        try:
                            score = (logits[token].item() - token_ranges[token][0]) / abs(token_ranges[token][1] - token_ranges[token][0])
                        except:
                            pass
                        scores.append({
                            "pos": pos_i,
                            "token": format_token(tokenizer, token),
                            "score": score,
                            "prob": partial_probs[i].item(),
                            "logit": logits[token].item(),
                            "logitdiff": logits[token].item() - logits[tokens[1 - i]].item(),
                            "layer": layer_i,
                            "label": label,
                            "base_label": base_label,
                        })
        
        # plot
        df = pd.DataFrame(scores)
        ticks = list(range(len(test.input_ids[0])))
        labels = [format_token(tokenizer, test.input_ids[0][i]) for i in ticks]
        plot = (ggplot(df, aes(x="pos", y="layer", fill="prob"))
                + facet_grid("label ~ token") + geom_tile()
                + scale_x_continuous(breaks=ticks, labels=labels)
                + scale_fill_gradient2(low="purple", high="orange", mid="white", midpoint=0.5)
                + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
        plot.save("figs/das/prob_per_pos.pdf")
        
        # plot
        plot = (ggplot(df, aes(x="pos", y="layer", fill="logitdiff"))
                + facet_grid("label ~ token") + geom_tile()
                + scale_x_continuous(breaks=ticks, labels=labels)
                + scale_fill_gradient2(low="purple", high="orange", mid="white", midpoint=0)
                + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
        plot.save("figs/das/logitdiff_per_pos.pdf")

        # print avg score per pos, token
        for pos_i in range(1, len(test.input_ids[0])):
            print(f"pos {pos_i}: {format_token(tokenizer, test.input_ids[0][pos_i])}")
            for label in [" he", " she"]:
                print(label)
                df = pd.DataFrame(scores)
                df = df[df["pos"] == pos_i]
                df = df[df["label"] == label]
                df = df.groupby(["label", "base_label", "token"]).mean(numeric_only=True)
                df = df.reset_index()

                # convert df to dict where key is label
                data = {}
                for _, row in df.iterrows():
                    data[row["token"]] = (row["score"], row["prob"], row["logit"])
                
                # print
                for label in data:
                    print(f"{label:>5}: {data[label][0]:>15.3%} {data[label][1]:>15.3%}")
            print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num_dims", type=int, default=-1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--grad_steps", type=int, default=4)
    args = parser.parse_args()
    experiment(**vars(args))


if __name__ == "__main__":
    main()
