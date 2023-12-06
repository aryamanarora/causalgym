import json
from datasets import Dataset
from transformers import AutoTokenizer
import random

def load_data():
    """Load data templates."""
    data = json.load(open("data/data.json", "r"))
    return data

def fill_variables(template, vars, label_var, label, other_label):
    """Fill variables in a template sentence."""
    base, src = template, template
    for var in vars:
        if var == label_var:
            base = base.replace(f"{{{var}}}", random.choice(vars[label_var][label]))
            src = src.replace(f"{{{var}}}", random.choice(vars[label_var][other_label]))
        else:
            val = random.choice(vars[var])
            base = base.replace(f"{{{var}}}", val)
            src = src.replace(f"{{{var}}}", val)
    return base, src


def make_data(tokenizer, experiment, batch_size, batches, device):
    """Make data for an experiment."""
    # load data
    data = load_data()[experiment]
    label_var = data["label"]
    vars = data["variables"]
    label_opts = list(vars[label_var].keys())

    # remove too long label vars
    for key in label_opts:
        vars[label_var][key] = [name for name in vars[label_var][key] if len(tokenizer(name)['input_ids']) == 1]
    print(vars[label_var])
    
    # make batches
    result = []
    for batch in range(batches):
        base, src, labels, base_labels, pos_i = [], [], [], [], []

        # make sents
        for _ in range(batch_size):
            template = random.choice(data["templates"])
            label = random.choice(label_opts)
            other_label = random.choice(label_opts)
            while other_label == label:
                other_label = random.choice(label_opts)
            base_i, src_i = fill_variables(template, vars, label_var, label, other_label)
            base.append(base_i)
            src.append(src_i)
            labels.append(other_label)
            base_labels.append(label)

        # tokenize
        pair = [
            tokenizer(base, return_tensors="pt", padding=True).to(device),
            tokenizer(src, return_tensors="pt", padding=True).to(device),
        ]

        # calculate final positions
        last_token_indices = pair[0].attention_mask.sum(1) - 1
        for i in range(batch_size):
            pos_i.append([1, last_token_indices[i].item()])
        
        # return
        labels = tokenizer(labels, return_tensors="pt").input_ids.to(device).reshape(-1)
        base_labels = tokenizer(base_labels, return_tensors="pt").input_ids.to(device).reshape(-1)
        result.append((pair, labels, base_labels, [pos_i,]))
    return result, label_opts

def test():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.pad_token = tokenizer.eos_token
    device = "cpu"
    data = make_data(tokenizer, "gender", 10, 1, device)
    print(data)

if __name__ == "__main__":
    test()