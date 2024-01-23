"""
Check if a model produces the expected output for a task.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import Dataset, list_datasets
from utils import WEIGHTS, MODELS, top_vals, format_token, get_last_token
import argparse
from tqdm import tqdm
import json

@torch.no_grad()
def benchmark(model=None, task=None, debug=False, rank=False):

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
        gpt.eval()
        # print model dtype
        print(f"{model:<30} {gpt.dtype}")

        # make data
        for dataset in datasets:
            data_source = Dataset.load_from(dataset)
            trainset = data_source.sample_batches(tokenizer, 4, 100, device, strategy="last", seed=42)
            count, correct = 0, 0
            probs = {}

            for batch in tqdm(trainset):
                # vars
                base_label = batch.base_labels
                src_label = batch.src_labels
                base_type = batch.base_types
                src_type = batch.src_types

                # inference
                base_output = gpt(**batch.base)
                src_output = gpt(**batch.src)
                base_logits = get_last_token(base_output.logits, batch.base['attention_mask'])
                src_logits = get_last_token(src_output.logits, batch.src['attention_mask'])

                # check for batch accuracy
                for i in range(4):
                    base_probs = torch.softmax(base_logits[i], dim=-1)
                    src_probs = torch.softmax(src_logits[i], dim=-1)
                    if base_probs[base_label[i]] > base_probs[src_label[i]] and src_probs[src_label[i]] > src_probs[base_label[i]]:
                        correct += 1
                    if debug:
                        print(base_probs[base_label[i]] > base_probs[src_label[i]] and src_probs[src_label[i]] > src_probs[base_label[i]])
                        print(tokenizer.decode(batch.base['input_ids'][i]))
                        top_vals(tokenizer, base_probs, n=5, highlight=[base_label[i], src_label[i]])
                        print(tokenizer.decode(batch.src['input_ids'][i]))
                        top_vals(tokenizer, src_probs, n=5, highlight=[base_label[i], src_label[i]])
                        input()
                    if count == 0:
                        probs[base_type[i]] = base_probs
                        probs[src_type[i]] = src_probs
                    else:
                        probs[base_type[i]] += base_probs
                        probs[src_type[i]] += src_probs
                    count += 1

            # store stats
            data.append({
                "model": model,
                "dataset": dataset,
                "count": count,
                "correct": correct,
                "iia": correct / count,
                "parameters": gpt.num_parameters(),
            })
            print(f"{dataset:<30} {correct / count:>10.2%} ({correct} / {count})")
            if rank:
                for k, v in probs.items():
                    probs[k] = (v / count)
                    print(k.upper())
                    top_vals(tokenizer, probs[k], n=10)
                    print('---')
                print("DIFF")
                top_vals(tokenizer, list(probs.values())[1] - list(probs.values())[0], n=10)
                print('---')
                top_vals(tokenizer, list(probs.values())[0] - list(probs.values())[1], n=10)
                print('---')
    
    # save data
    with open("logs/benchmark.json", "w") as f:
        json.dump(data, f)
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rank", action="store_true")
    args = parser.parse_args()
    benchmark(**vars(args))

if __name__ == "__main__":
    main()