from data import list_datasets
from das import experiment
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import WEIGHTS
import torch

def run_command(tokenizer: AutoTokenizer, gpt: AutoModelForCausalLM, model_name: str, dataset: str, hparam_non_das: bool = False):
    # command = f"python das.py --model EleutherAI/pythia-70m --intervention {method} --dataset {dataset} --position each --num-tokens 1 --num-dims 1 --steps {steps}"
    print(dataset)
    experiment(
        model=model_name,
        dataset=dataset,
        steps=100,
        eval_steps=25,
        grad_steps=1,
        batch_size=4,
        intervention_site="block_output",
        strategy="last",
        lr=5e-3,
        only_das=False,
        hparam_non_das=hparam_non_das,
        tokenizer=tokenizer,
        gpt=gpt,
    )

def main(model: str, hparam_non_das: bool = False):
    # load model + tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)

    # run commands
    datasets = [d for d in list_datasets() if d.startswith("syntaxgym/")]
    for dataset in datasets:
        run_command(tokenizer, gpt, model, dataset, hparam_non_das)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--hparam_non_das", action="store_true")
    args = parser.parse_args()
    main(**vars(args))