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
def benchmark(model=None, task=None, debug=False):

    # get models, data
    if model is None:
        models = reversed(MODELS)
    else:
        models = [model]
    datasets = [dataset for dataset in list_datasets() if dataset.startswith(f"syntaxgym/{task if task is not None else ''}")]
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
            trainset, _ = make_data(tokenizer, dataset, 4, 100, -1, device)
            count, correct = 0, 0
            probs_base, probs_src = [], []

            for (pair, src_label, base_label, _, _, _) in tqdm(trainset):
                # inference
                base_output = gpt(**pair[0])
                src_output = gpt(**pair[1])
                base_logits = get_last_token(base_output.logits, pair[0].attention_mask)
                src_logits = get_last_token(src_output.logits, pair[1].attention_mask)

                # check for batch accuracy
                for i in range(4):
                    base_probs = torch.softmax(base_logits[i], dim=-1)
                    src_probs = torch.softmax(src_logits[i], dim=-1)
                    if base_probs[base_label[i]] > base_probs[src_label[i]] and src_probs[src_label[i]] > src_probs[base_label[i]]:
                        correct += 1
                    if debug:
                        print(base_probs[base_label[i]] > base_probs[src_label[i]] and src_probs[src_label[i]] > src_probs[base_label[i]])
                        print(tokenizer.decode(pair[0].input_ids[i]))
                        top_vals(tokenizer, base_probs, n=5, highlight=[base_label[i], src_label[i]])
                        print(tokenizer.decode(pair[1].input_ids[i]))
                        top_vals(tokenizer, src_probs, n=5, highlight=[base_label[i], src_label[i]])
                        input()

                    count += 1

            # store stats
            data.append({
                "dataset": dataset,
                "count": count,
                "correct": correct,
                "iia": correct / count,
                "parameters": gpt.num_parameters(),
            })
            print(f"{dataset:<30} {correct / count:>10.2%} ({correct} / {count})")
    
    # save data
    with open("logs/benchmark.json", "w") as f:
        json.dump(data, f)
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    benchmark(**vars(args))

if __name__ == "__main__":
    main()