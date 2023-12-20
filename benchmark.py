"""
Check if a model produces the expected output for a task.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import make_data, list_datasets
from utils import WEIGHTS, top_vals, format_token
import argparse
from tqdm import tqdm

def benchmark(model):
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

    # make data
    datasets = list_datasets()
    for dataset in datasets:
        trainset, labels = make_data(tokenizer, dataset, 1, 200, -1, device)
        count, correct = 0, 0
        probs_base, probs_src = [], []

        for (pair, src_label, base_label, pos_base) in tqdm(trainset):
            for pair_i in range(2):
                # inference
                logits = gpt(**pair[pair_i]).logits[0, -1]
                probs = torch.softmax(logits, dim=-1)

                # print
                # print(tokenizer.decode(pair[pair_i].input_ids[0]))
                # top_vals(tokenizer, probs, [base_label])

                # accuracy
                if probs[base_label] > probs[src_label]:
                    correct += 1
                probs_base.append(probs[base_label].item())
                probs_src.append(probs[src_label].item())
                # else:
                #     print(tokenizer.decode(pair[pair_i].input_ids[0]))
                #     top_vals(tokenizer, probs, [base_label])

                count += 1

                # swap labels
                src_label, base_label = base_label, src_label

        print(f"{dataset:<30} {correct / count:>10.2%} ({correct}/{count})")
        print(f"{dataset:<30} {sum(probs_base) / len(probs_base):>10.2%} {sum(probs_src) / len(probs_src):>10.2%}")
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    args = parser.parse_args()
    benchmark(**vars(args))

if __name__ == "__main__":
    main()