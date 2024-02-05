from numpy import add
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
from utils import WEIGHTS
from data import Dataset
from eval import eval, augment_data
from train import train_das, train_feature_direction, method_mapping, additional_method_mapping
import datetime
import json
from typing import Union

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
    strategy: str,
    lr: float,
    only_das: bool=False,
    hparam_non_das: bool=False,
    das_label: str=None,
    tokenizer: Union[AutoTokenizer, None]=None,
    gpt: Union[AutoModelForCausalLM, None]=None,
):
    """Run a feature-finding experiment."""

    # load model
    total_data = []
    diff_vectors = []
    NOW = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
    if gpt is None:
        weight_type = WEIGHTS.get(model, torch.float16) if device == "cuda:0" else torch.float32
        gpt = GPTNeoXForCausalLM.from_pretrained(
            model,
            revision="main",
            torch_dtype=weight_type,
            use_flash_attention_2=(weight_type in [torch.bfloat16, torch.float16] and device == "cuda:0"),
        ).to(device)
    print(model, gpt.config.num_hidden_layers)
    gpt.eval()

    # make dataset, ensuring examples in trainset are not in evalset
    data_source = Dataset.load_from(dataset)
    trainset = data_source.sample_batches(tokenizer, batch_size, steps, device, seed=42)
    discard = set()
    for batch in trainset:
        for pair in batch.pairs:
            discard.add(''.join(pair.base))
    evalset = data_source.sample_batches(tokenizer, batch_size, 25, device, seed=420, discard=discard)
    
    # methods
    methods = list(method_mapping.keys())
    if hparam_non_das:
        methods.extend(list(additional_method_mapping.keys()))
    print(methods)
    
    # entering train loops
    for pos_i in range(data_source.first_var_pos, data_source.length):
        if trainset[0].compute_pos(strategy)[0][0][pos_i][0] == -1:
            continue

        # per-layer training loop
        iterator = range(gpt.config.num_hidden_layers)
        for layer_i in iterator:
            print(f"position {pos_i} ({data_source.span_names[pos_i]}), layer {layer_i}")
            data = []

            # vanilla intervention
            if strategy != "all":
                intervenable_config = intervention_config(
                    intervention_site, pv.VanillaIntervention, layer_i, 0
                )
                intervenable = IntervenableModel(intervenable_config, gpt)
                intervenable.set_device(device)
                intervenable.disable_model_gradients()

                more_data, summary, _ = eval(intervenable, evalset, layer_i, pos_i, strategy)
                intervenable._cleanup_states()
                data.extend(augment_data(more_data, {"method": "vanilla", "step": -1}))
                print(f"vanilla: {summary}")

            # DAS intervention
            intervenable_config = intervention_config(
                intervention_site,
                pv.LowRankRotatedSpaceIntervention if strategy != "all" else PooledLowRankRotatedSpaceIntervention,
                layer_i, 1
            )
            intervenable = IntervenableModel(intervenable_config, gpt)
            intervenable.set_device(device)
            intervenable.disable_model_gradients()

            _, more_data, activations, eval_activations, diff_vector = train_das(
                intervenable, trainset, evalset, layer_i, pos_i, strategy,
                eval_steps, grad_steps, lr=lr)
            diff_vectors.append({"method": "das" if das_label is None else das_label,
                                 "layer": layer_i, "pos": pos_i, "vec": diff_vector})
            data.extend(more_data)
            
            # test other methods
            if not only_das:
                for method in methods:
                    try:
                        more_data, summary, diff_vector = train_feature_direction(
                            method, intervenable, activations, eval_activations,
                            evalset, layer_i, pos_i, strategy, intervention_site
                        )
                        print(f"{method}: {summary}")
                        diff_vectors.append({"method": method, "layer": layer_i, "pos": pos_i, "vec": diff_vector})
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
            "strategy": strategy,
            "lr": lr,
            "span_names": data_source.span_names,
        },
        "data": total_data,
        "vec": diff_vectors,
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
    parser.add_argument("--strategy", type=str, default="last")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--only-das", action="store_true")
    parser.add_argument("--hparam-non-das", action="store_true")
    parser.add_argument("--das-label", type=str, default=None)
    args = parser.parse_args()
    print(vars(args))
    experiment(**vars(args))


if __name__ == "__main__":
    main()
