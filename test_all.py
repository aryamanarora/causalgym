from data import list_datasets
from das import experiment
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import WEIGHTS
import torch
import os

def run_command(
    tokenizer: AutoTokenizer,
    gpt: AutoModelForCausalLM,
    model_name: str,
    dataset: str,
    lr: float,
    only_das: bool,
    hparam_non_das: bool,
    das_label: str,
    revision: str,
    folder: str,
    manipulate: str,
):
    # command = f"python das.py --model EleutherAI/pythia-70m --intervention {method} --dataset {dataset} --position each --num-tokens 1 --num-dims 1 --steps {steps}"
    print(dataset)
    experiment(
        model=model_name,
        dataset=dataset,
        steps=100,
        eval_steps=100,
        grad_steps=1,
        batch_size=4,
        intervention_site="block_output",
        strategy="last",
        lr=lr,
        only_das=only_das,
        hparam_non_das=hparam_non_das,
        das_label=das_label,
        revision=revision,
        log_folder=folder,
        manipulate=manipulate,
        tokenizer=tokenizer,
        gpt=gpt,
    )

def main(
    model: str, lr: float=5e-3, hparam_non_das: bool=False, only_das: bool=False,
    das_label: str=None, start: int=None, end: int=None, folder: str="das", revision: str="main",
    manipulate: str=False):

    # load model + tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)

    # run commands
    datasets = [d for d in list_datasets() if d.startswith("syntaxgym/")]
    print(len(datasets))

    # start/end
    if start is None:
        start = 0
    if end is None:
        end = len(datasets)

    # make folder
    if not os.path.exists(f"logs/{folder}"):
        os.makedirs(f"logs/{folder}")

    for dataset in datasets[start:end]:
        run_command(tokenizer, gpt, model, dataset, lr, only_das, hparam_non_das, das_label, revision, folder, manipulate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--only-das", action="store_true")
    parser.add_argument("--hparam_non_das", action="store_true")
    parser.add_argument("--das-label", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--folder", type=str, default="das")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--manipulate", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))