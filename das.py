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
from utils import MODELS, WEIGHTS, get_last_token, format_token
from data import make_data
from eval import calculate_loss, eval, eval_sentence
from train import *
import plot
import datetime
import json

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.alignable_base import AlignableModel
from interventions import *

def experiment(
    model: str,
    dataset: str,
    steps: int,
    intervention: str,
    num_dims: int,
    warmup: bool,
    eval_steps: int,
    grad_steps: int,
    batch_size: int,
    num_tokens: int,
    position: str,
    intervention_site: str,
    store_weights: bool,
    do_swap: bool=True,
    test_sentence: bool=False,
    plot_now: bool=False
):
    """Run a feature-finding experiment."""

    # load model
    NOW = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)
    print(gpt.config.num_hidden_layers)

    # setup
    if intervention == "vanilla":
        num_dims = 0
    elif intervention != "das":
        num_dims = None

    # make das subdir
    if not os.path.exists("figs/das"):
        os.makedirs("figs/das")
    if not os.path.exists("figs/das/steps"):
        os.makedirs("figs/das/steps")
    if not os.path.exists("logs/das"):
        os.makedirs("logs/das")
    
    # clear files from figs/das/steps
    for file in os.listdir("figs/das/steps"):
        os.remove(os.path.join("figs/das/steps", file))

    # intervene on each layer
    data, weights = [], []
    layer_objs = {}
    
    # entering train loops
    max_loop, pos_i = 2, 1
    while pos_i < (max_loop if position == "each" else 1):

        # train and eval sets
        trainset, labels = make_data(tokenizer, dataset, batch_size, steps, num_tokens, device, position=position if position != "each" else pos_i, seed=42)
        evalset, _ = make_data(tokenizer, dataset, batch_size, 20, num_tokens, device, position=position if position != "each" else pos_i, seed=420)
        max_loop = len(trainset[0].pair[0].input_ids[0])

        # tokens to log
        tokens = tokenizer.encode("".join(labels))

        # per-layer training loop
        iterator = tqdm(range(gpt.config.num_hidden_layers))
        for layer_i in iterator:
            print(f"position {pos_i} of {max_loop}, layer {layer_i}")

            # set up alignable model
            alignable_config = intervention_config(
                type(gpt), intervention_site, layer_i, num_dims
            )
            alignable = AlignableModel(alignable_config, gpt)
            alignable.set_device(device)
            alignable.disable_model_gradients()

            # training
            if intervention == "das":
                _, more_data, more_weights = train_das(
                    alignable, tokenizer, trainset, evalset, layer_i,
                    pos_i, num_dims, steps, warmup, eval_steps, grad_steps,
                    store_weights, tokens
                )
                weights.extend(more_weights)
            elif intervention == "vanilla":
                more_data, more_stats = eval(alignable, tokenizer, evalset,
                                             layer_i, 0, tokens, num_dims)
                iterator.set_postfix(more_stats)
            elif intervention in ["mean_diff", "kmeans", "probe", "probe_sklearn", "pca"]:
                more_data, more_stats = train_feature_direction(
                    intervention, alignable, tokenizer, trainset, evalset,
                    layer_i, pos_i, intervention_site, tokens
                )
                iterator.set_postfix(more_stats)
                
            # store obj
            layer_objs[layer_i] = alignable
            data.extend(more_data)
        
        pos_i += 1

    # make data dump
    short_dataset_name = dataset.split('/')[-1]
    short_model_name = model.split('/')[-1]
    filedump = {
        "metadata": {
            "model": model,
            "dataset": dataset,
            "intervention": intervention,
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
            "test_sentence": test_sentence,
        },
        "weights": weights,
        "data": data
    }

    # log
    log_file = f"logs/das/{short_model_name}__{short_dataset_name}__{NOW}.json"
    print(f"logging to {log_file}")
    with open(log_file, "w") as f:
        json.dump(filedump, f)

    # print plots
    if plot_now:
        df = pd.DataFrame(data)
        plot.plot_bounds(df, f"{short_dataset_name}, {short_model_name}: intervention boundary")
        plot.plot_label_loss(df, f"{short_dataset_name}, {short_model_name}: per-label loss")
        plot.plot_label_prob(df, f"{short_dataset_name}, {short_model_name}: per-label probs")
        plot.plot_label_logit(df, f"{short_dataset_name}, {short_model_name}: per-label logits")

        # cosine sim of learned directions plot
        if num_dims == 1:
            plot.plot_das_cos_sim(layer_objs, f"{short_dataset_name}, {short_model_name}: cosine similarity of learned directions")
        
        # iia per position/layer plot
        if position == "each":
            sentence = evalset[0].pair[0].input_ids[0]
            other_sentence = evalset[0].pair[1].input_ids[0]
            labels = []
            for i in range(len(sentence)):
                if sentence[i] != other_sentence[i]:
                    labels.append(format_token(tokenizer, sentence[i]) + ' / ' + format_token(tokenizer, other_sentence[i]))
                else:
                    labels.append(format_token(tokenizer, sentence[i]))
            plot.plot_pos_iia(df, f"{short_dataset_name}, {short_model_name}: position iia", sentence=labels)

        # make gif of files in figs/das/steps
        os.system("convert -delay 100 -loop 0 figs/das/steps/*prob_per_pos.png figs/das/prob_steps.gif")
        # os.system("convert -delay 100 -loop 0 figs/das/steps/*val_per_pos.png figs/das/val_steps.gif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="gender_basic")
    parser.add_argument("--intervention", type=str, default="das")
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--num-dims", type=int, default=-1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--grad-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=-1)
    parser.add_argument("--position", type=str, default="all")
    parser.add_argument("--intervention-site", type=str, default="block_output")
    parser.add_argument("--store-weights", action="store_true")
    parser.add_argument("--plot-now", action="store_true")
    args = parser.parse_args()
    print(vars(args))
    experiment(**vars(args))


if __name__ == "__main__":
    main()
