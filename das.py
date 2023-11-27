import torch
import csv
import random
from utils import MODELS, WEIGHTS, Sentence, get_options, make_sentence
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import sys
import pandas as pd
from tqdm import tqdm
import argparse
from plotnine import ggplot, geom_point, geom_label, geom_tile, geom_smooth, aes, facet_wrap, theme, element_text, \
                     facet_grid, geom_bar, geom_hline, scale_y_log10, ggtitle
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse
from typing import List
from utils import names
import argparse

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import embed_to_distrib, top_vals, format_token, lsm, sm, count_parameters
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention, RotatedSpaceIntervention, BoundlessRotatedSpaceIntervention
from models.gpt_neox.modelings_alignable_gpt_neox import create_gpt_neox
from umap import UMAP

def rotated_space_intervention(num_dims):
    def func(args, proj_dim):
        intervention = RotatedSpaceIntervention(args)
        intervention.set_interchange_dim(num_dims)
        return intervention
    return func

def intervention_config(model_type, intervention_type, layer, num_dims):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=BoundlessRotatedSpaceIntervention
    )
    return alignable_config

def experiment(model="EleutherAI/pythia-70m", steps=1000):
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32
    ).to(device)

    def make_pair():
        gender1 = random.choice(["he", "she"])
        gender2 = random.choice(["he", "she"])
        he = random.choice(names[gender1])
        she = random.choice(names[gender2])
        completions = ["is tired", "went home", "walked", "ran", "works there", "joined the army"]
        completion = random.choice(completions)
        pair = (
            tokenizer(f"<|endoftext|>{he} {completion} because", return_tensors="pt").to(device),
            tokenizer(f"<|endoftext|>{she} {completion} because", return_tensors="pt").to(device),
        )
        return pair, " " + gender2

    # tokenize
    tokens = tokenizer.encode(" she he") # token we want to maximize the probability of

    # evalset
    evalset = []
    for i in range(20):
        evalset.append(make_pair())

    # intervene on each layer
    # only intervening on layer 0, pos 1, dim 1
    data = []

    # for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
    for layer_i in [0]:

        # for pos_i in range(1, len(base.input_ids[0])):
        for pos_i in [1]:
            
            # how many dims to intervene on
            for num_dims in [1]:

                print(layer_i, pos_i, num_dims)
            
                # set up alignable model
                alignable_config = intervention_config(type(gpt), "block_output", layer_i, num_dims)
                alignable = AlignableModel(alignable_config, gpt)
                alignable.set_device(device)
                alignable.disable_model_gradients()

                # optimizer
                t_total = steps
                warm_up_steps = 0.1 * t_total
                optimizer_params = []
                for k, v in alignable.interventions.items():
                    optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
                    optimizer_params += [{'params': v[0].intervention_boundaries, 'lr': 1e-2}]
                    
                print("model trainable parameters: ", count_parameters(alignable.model))
                print("intervention trainable parameters: ", alignable.count_parameters())

                optimizer = torch.optim.Adam(
                    optimizer_params,
                    lr=1e-3
                )
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warm_up_steps,
                    num_training_steps=t_total
                )

                gradient_accumulation_steps = 4
                total_step = 0
                temperature_start = 50.0
                temperature_end = 0.1
                temperature_schedule = torch.linspace(
                    temperature_start, temperature_end, t_total
                ).to(torch.bfloat16).to(device)
                alignable.set_temperature(temperature_schedule[total_step])

                def calculate_loss(logits, label):
                    shift_logits = logits[..., :, :].contiguous()
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits[0, -1].softmax(-1)
                    shift_labels = torch.tensor(label)
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
        
                    for k, v in alignable.interventions.items():
                        boundary_loss = 1. * v[0].intervention_boundaries.sum()
                    loss += boundary_loss
                    
                    return loss
                
                iterator = tqdm(range(t_total))
                for step in iterator:

                    # make pair
                    pair, label = make_pair()

                    # inference
                    _, counterfactual_outputs = alignable(
                        pair[0],
                        [pair[1]],
                        {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
                    )
                    
                    # loss and backprop
                    loss = calculate_loss(
                        counterfactual_outputs.logits,
                        tokenizer.encode(label)[0]
                    )
                    loss_str = round(loss.item(), 2)

                    # print stats
                    stats = {'loss': loss_str}
                    distrib = sm(counterfactual_outputs.logits)[0, -1]
                    for tok in tokens:
                        prob = distrib[tok].item()
                        stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
                    hidden_state_size = gpt.config.hidden_size
                    for k, v in alignable.interventions.items():
                        stats["bound"] = f"{v[0].intervention_boundaries.sum() * v[0].embed_dim:.3f}"
                    iterator.set_postfix(stats)

                    # gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                    if total_step % gradient_accumulation_steps == 0:
                        if not (gradient_accumulation_steps > 1 and total_step == 0):
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            alignable.set_zero_grad()
                            alignable.set_temperature(temperature_schedule[total_step])

                    # eval
                    if step % 20 == 0:

                        with torch.no_grad():
                            boundary = None
                            for k, v in alignable.interventions.items():
                                boundary = v[0].intervention_boundaries.sum() * v[0].embed_dim
                            boundary = boundary.item()

                            for pair, label in evalset:
                                _, counterfactual_outputs = alignable(
                                    pair[0],
                                    [pair[1]],
                                    {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
                                )
                                loss = calculate_loss(
                                    counterfactual_outputs.logits,
                                    tokenizer.encode(label)[0]
                                )
                                distrib = sm(counterfactual_outputs.logits)[0, -1]
                                for tok in tokens:
                                    prob = distrib[tok].item()
                                    stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
                                    data.append({
                                        "step": step,
                                        "label": label,
                                        "loss": loss.item(),
                                        "token": format_token(tokenizer, tok),
                                        "prob": prob,
                                        "bound": boundary,
                                    })

                    total_step += 1
    
    # print plots
    df = pd.DataFrame(data)
    
    plot = (ggplot(df, aes(x="step", y="bound")) + geom_smooth() + ggtitle("intervention boundary"))
    plot.save("figs/bound.pdf")

    plot = (ggplot(df, aes(x="step", y="loss", color="factor(label)")) + geom_smooth() + ggtitle("per-label loss"))
    plot.save("figs/loss.pdf")

    plot = (ggplot(df, aes(x="step", y="prob", color="factor(label)")) + facet_grid("~token") + geom_smooth() + ggtitle("per-label probs"))
    plot.save("figs/prob.pdf")

    # test probe on a sentence
    test = tokenizer("<|endoftext|>He is my girlfriend's brother and he wants to be a nurse.", return_tensors="pt")
    neutrals = [tokenizer("<|endoftext|>John fell because", return_tensors="pt"), tokenizer("<|endoftext|>Jane fell because", return_tensors="pt")]
    base_logits = [{}, {}]

    for i in range(len(neutrals)):
        logits = gpt(**neutrals[i].to(device)).logits
        for token in tokens:
            base_logits[i][token] = logits[0, -1, token].item()

    for pos_i in range(1, len(test.input_ids[0])):
        for i, neutral in enumerate(neutrals):
            _, counterfactual_outputs = alignable(
                neutral,
                [test],
                {"sources->base": ([[[pos_i]]], [[[2]]])}
            )

            logits = counterfactual_outputs.logits[0, -1]
            for token in tokens:
                print(f"{pos_i:<5} {format_token(tokenizer, test.input_ids[0][pos_i]):<15} {format_token(tokenizer, token):<15} {logits[token].item() - base_logits[i][token]:>9.4f}")
        print()

    # # make df
    # df = pd.DataFrame(data)
    # df['layer'] = df['layer'].astype('int')
    # df['pos'] = df['pos'].astype('int')
    # df['prob'] = df['prob'].astype('float')

    # # plot
    # plot = (ggplot(df, aes(x="layer", y="pos")) + scale_y_reverse() + facet_grid("~token")
    #         + geom_tile(aes(fill="prob")) + scale_fill_cmap("Purples"))
    # print(plot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    experiment(args.model)

if __name__ == "__main__":
    main()