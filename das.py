from turtle import pos
import torch
import os
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import WEIGHTS
from data import Dataset
from eval import eval, augment_data
from train import train_das, train_feature_direction, method_to_class_mapping
import datetime
import json

from pyvene.models.intervenable_base import IntervenableModel
from interventions import *


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


def experiment(
    model: str,
    dataset: str,
    steps: int,
    eval_steps: int,
    grad_steps: int,
    batch_size: int,
    intervention_site: str,
):
    """Run a feature-finding experiment."""

    # load model
    total_data = []
    NOW = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)
    print(model, gpt.config.num_hidden_layers)

    # make dataset
    data_source = Dataset.load_from(dataset)
    trainset = data_source.sample_batches(tokenizer, batch_size, steps, device, strategy="last", seed=42, model=gpt)
    evalset = data_source.sample_batches(tokenizer, batch_size, 20, device, strategy="last", seed=420, model=gpt)
    
    # entering train loops
    for pos_i in range(data_source.length):
        if trainset[0].pos[0, 0, pos_i, 0] == -1:
            continue

        # per-layer training loop
        iterator = tqdm(range(gpt.config.num_hidden_layers))
        for layer_i in iterator:
            tqdm.write(f"position {pos_i} ({data_source.span_names[pos_i]}), layer {layer_i}")
            data = []

            # vanilla intervention
            intervenable_config = intervention_config(
                type(gpt), intervention_site, layer_i, 0
            )
            intervenable = IntervenableModel(intervenable_config, gpt)
            intervenable.set_device(device)
            intervenable.disable_model_gradients()

            more_data, summary = eval(intervenable, evalset, layer_i, pos_i)
            data.extend(augment_data(more_data, {"method": "vanilla", "step": -1}))
            tqdm.write(f"vanilla: {summary}")

            # DAS intervention
            intervenable_config = intervention_config(
                type(gpt), intervention_site, layer_i, 1
            )
            intervenable = IntervenableModel(intervenable_config, gpt)
            intervenable.set_device(device)
            intervenable.disable_model_gradients()

            _, more_data, activations = train_das(intervenable, trainset, evalset, layer_i, pos_i, eval_steps, grad_steps)
            data.extend(more_data)
            
            # test other methods
            for method in method_to_class_mapping.keys():
                try:
                    more_data, summary = train_feature_direction(
                        method, intervenable, activations, evalset,
                        layer_i, pos_i, intervention_site
                    )
                    tqdm.write(f"{method}: {summary}")
                    data.extend(more_data)
                except:
                    continue
            
            # store all data
            total_data.extend(augment_data(data, {"layer": layer_i, "pos": pos_i}))

    # make data dump
    short_dataset_name = dataset.split('/')[-1]
    short_model_name = model.split('/')[-1]
    filedump = {
        "metadata": {
            "model": model,
            "dataset": dataset,
            "steps": steps,
            "eval_steps": eval_steps,
            "grad_steps": grad_steps,
            "batch_size": batch_size,
            "intervention_site": intervention_site,
            "span_names": data_source.span_names,
        },
        "data": total_data
    }

    # log
    log_file = f"logs/das/{NOW}__{short_model_name}__{short_dataset_name}.json"
    print(f"logging to {log_file}")
    with open(log_file, "w") as f:
        json.dump(filedump, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="syntaxgym/agr_gender")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--grad-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--intervention-site", type=str, default="block_output")
    args = parser.parse_args()
    print(vars(args))
    experiment(**vars(args))


if __name__ == "__main__":
    main()
