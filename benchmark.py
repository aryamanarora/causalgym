"""
Check if a model produces the expected output for a task.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import make_data, list_datasets
from utils import WEIGHTS, MODELS, top_vals, format_token, get_last_token
import argparse
from tqdm import tqdm
import json

@torch.no_grad()
def benchmark(model=None):

    # get models, data
    if model is None:
        models = [model for model in MODELS if model.startswith("EleutherAI")]
    else:
        models = [model]
    datasets = list_datasets()
    data = []

    # benchmark
    for model in models:

        # load model
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        gpt = AutoModelForCausalLM.from_pretrained(
            model,
            revision="main",
            torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
        ).to(device)
        print(model)
        print(gpt.config.num_hidden_layers)
        gpt.eval()

        # make data
        for dataset in datasets:
            trainset, labels = make_data(tokenizer, dataset, 4, 50, -1, device)
            count, correct = 0, 0
            probs_base, probs_src = [], []

            for (pair, src_label, base_label, pos_base) in tqdm(trainset):
                for pair_i in range(2):
                    # inference
                    output = gpt(**pair[pair_i])
                    logits = get_last_token(output.logits, pair[0].attention_mask)

                    # accuracy
                    for i in range(4):
                        probs = torch.softmax(logits[i], dim=-1)
                        if probs[base_label[i]] > probs[src_label[i]]:
                            correct += 1
                        probs_base.append(probs[base_label[i]].item())
                        probs_src.append(probs[src_label[i]].item())

                        count += 1

                    # swap labels
                    src_label, base_label = base_label, src_label

            # store stats
            data.append({
                "dataset": dataset,
                "count": count,
                "correct": correct,
                "probs_base": sum(probs_base) / len(probs_base),
                "probs_src": sum(probs_src) / len(probs_src),
                "iia": correct / count,
                "parameters": gpt.num_parameters(),
            })
            print(f"{dataset:<30} {correct / count:>10.2%} ({correct}/{count})")
            print(f"{dataset:<30} {sum(probs_base) / len(probs_base):>10.2%} {sum(probs_src) / len(probs_src):>10.2%}")
    
    # save data
    with open("logs/benchmark.json", "w") as f:
        json.dump(data, f)
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    benchmark(**vars(args))

if __name__ == "__main__":
    main()