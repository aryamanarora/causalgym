import json
from datasets import Dataset
from transformers import AutoTokenizer
import random
import torch
from collections import defaultdict

random.seed(42)

def load_data():
    """Load data templates."""
    data = json.load(open("data/data.json", "r"))
    return data

def list_datasets():
    data = load_data()
    return list(data.keys())

def fill_variables(template, variables, num_tokens, label_var, label, other_label):
    """Fill variables in a template sentence."""
    base, src = template, template
    for var in variables:
        if var == label_var:
            base = base.replace(f"{{{var}}}", random.choice(variables[label_var][num_tokens][label]))
            src = src.replace(f"{{{var}}}", random.choice(variables[label_var][num_tokens][other_label]))
        else:
            val = random.choice(variables[var])
            base = base.replace(f"{{{var}}}", val)
            src = src.replace(f"{{{var}}}", val)
    return base, src


def make_data(tokenizer, experiment, batch_size, batches, num_tokens_limit=-1, device="cpu", position="all"):
    """Make data for an experiment."""
    # load data
    data = load_data()[experiment]
    label_var = data["label"]
    variables = data["variables"]

    # group by # tokens
    grouped_by_tokens = defaultdict(lambda: defaultdict(list))
    for label_opt in variables[label_var]:
        for option in variables[label_var][label_opt]:
            grouped_by_tokens[len(tokenizer(option)["input_ids"])][label_opt].append(option)

    # delete tokens that lack all options
    original_num_options = len(variables[label_var])
    for num_tokens in list(grouped_by_tokens.keys()):
        if num_tokens_limit != -1 and num_tokens != num_tokens_limit:
            del grouped_by_tokens[num_tokens]
        elif len(grouped_by_tokens[num_tokens]) < original_num_options:
            del grouped_by_tokens[num_tokens]

    # make token options
    variables[label_var] = grouped_by_tokens
    token_opts = list(variables[label_var].keys())
    # print(variables[label_var])
    
    # make batches
    result = []
    for batch in range(batches):
        base, src, src_labels, base_labels, pos_i = [], [], [], [], []

        # make sents
        for _ in range(batch_size):
            template = random.choice(data["templates"])
            num_tokens = random.choice(token_opts)
            label_opts = list(variables[label_var][num_tokens].keys())
            label = random.choice(label_opts)
            other_label = random.choice(label_opts)
            while other_label == label:
                other_label = random.choice(label_opts)
            base_i, src_i = fill_variables(template, variables, num_tokens, label_var, label, other_label)
            base.append(base_i)
            src.append(src_i)
            src_labels.append(other_label)
            base_labels.append(label)

        # tokenize
        pair = [
            tokenizer(base, return_tensors="pt", padding=True).to(device),
            tokenizer(src, return_tensors="pt", padding=True).to(device),
        ]

        # positions to intervene on
        if position == "all":
            shape = pair[0].input_ids.shape
            pos_i = torch.arange(shape[1]).repeat(shape[0], 1).unsqueeze(0)
            pos_i = pos_i.tolist()
        elif position == "label":
            not_matching = pair[0].input_ids != pair[1].input_ids
            non_matching_indices = [torch.nonzero(pair[0].input_ids[p] != pair[1].input_ids[p], as_tuple=False).reshape(-1) for p in range(batch_size)]
            max_length = max(len(indices) for indices in non_matching_indices)
            padded_indices = [torch.nn.functional.pad(indices, (0, max_length - len(indices)), mode='constant', value=0) for indices in non_matching_indices]
            padded_indices_2d_tensor = torch.stack(padded_indices).unsqueeze(0)
            pos_i = padded_indices_2d_tensor.tolist()
        elif position == "first+last":
            last_token_indices = pair[0].attention_mask.sum(1) - 1
            for i in range(batch_size):
                pos_i.append([1, last_token_indices[i]])
            pos_i = [pos_i,]
        elif position == "last":
            last_token_indices = pair[0].attention_mask.sum(1) - 1
            for i in range(batch_size):
                pos_i.append([last_token_indices[i]])
            pos_i = [pos_i,]
        else:
            raise ValueError(f"Invalid position {position}")

        
        # return
        src_labels = tokenizer(src_labels, return_tensors="pt").input_ids.to(device).reshape(-1)
        base_labels = tokenizer(base_labels, return_tensors="pt").input_ids.to(device).reshape(-1)
        result.append((pair, src_labels, base_labels, pos_i))
    
    return result, label_opts

def test():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.pad_token = tokenizer.eos_token
    device = "cpu"
    data = make_data(tokenizer, "gender", 3, 2, -1, device, "label")
    print(data)

if __name__ == "__main__":
    test()