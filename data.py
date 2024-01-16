import json
from datasets import Dataset
from transformers import AutoTokenizer
import random
import torch
from collections import defaultdict, namedtuple, Counter
import json
import glob

random.seed(42)
Batch = namedtuple("Batch", ["pair", "src_labels", "base_labels", "pos_i", "src_label_overall", "base_label_overall"])
LabelSet = namedtuple("LabelSet", ["label_var", "num_tokens"])


def load_data(template_file):
    """Load data templates."""
    data = json.load(open(f"data/templates/{template_file}.json", "r"))
    return data


def list_datasets():
    keys = []
    for file in glob.glob("data/templates/*.json"):
        prefix = file.split('/')[-1].split('.')[0]
        keys.extend([prefix + '/' + x for x in load_data(prefix).keys()])
    return keys


def fill_variables(template, variables, grouped_by_tokens, label_vars, label, other_label):
    """Fill variables in a template sentence."""
    base, src = template, template
    for var in variables:
        if var in label_vars:
            label_sets = [label_set for label_set in grouped_by_tokens if label_set.label_var == var]
            label_set = random.choice(label_sets)
            base = base.replace(f"{{{var}}}", random.choice(grouped_by_tokens[label_set][label]))
            src = src.replace(f"{{{var}}}", random.choice(grouped_by_tokens[label_set][other_label]))
        else:
            val = random.choice(variables[var])
            base = base.replace(f"{{{var}}}", val)
            src = src.replace(f"{{{var}}}", val)
    return base, src


def make_data(tokenizer, experiment, batch_size, batches, num_tokens_limit=-1, device="cpu", position="all", seed=42):
    """Make data for an experiment."""

    # load data
    random.seed(seed)
    template_file = "data"
    if '/' in experiment:
        template_file, experiment = experiment.split('/')
    data = load_data(template_file)[experiment]

    # get vars
    label_vars = data["label"]
    if isinstance(label_vars, str):
        label_vars = [label_vars]
    variables = data["variables"]
    labels = data["labels"] if "labels" in data else {label_opt: [label_opt] for label_opt in variables[label_vars[0]]}
    labels = {label_opt: [" " + label for label in labels[label_opt]] for label_opt in labels}
    all_labels = list(set([label for label_opt in labels for label in labels[label_opt]]))
    data["templates"] = ["<|endoftext|>" + template for template in data["templates"]]

    # count possible
    possible = 1
    per_label = defaultdict(lambda: 1)
    for var in variables:
        if isinstance(variables[var], dict):
            for opt in variables[var]:
                per_label[opt] *= len(set(variables[var][opt]))
        else:
            possible *= len(set(variables[var]))
    total = sum(list(per_label.values())) * possible
    print(f"{experiment} possible:", total)

    # group by # tokens
    grouped_by_tokens = defaultdict(lambda: defaultdict(list))
    for label_var in label_vars:
        for label_opt in variables[label_var]:
            for option in variables[label_var][label_opt]:
                token = ' ' + option if data["label_prepend_space"] else option
                grouped_by_tokens[LabelSet(label_var, len(tokenizer(token)["input_ids"]))][label_opt].append(option)

    # apply limits
    original_num_options = len(labels)
    for label_set in list(grouped_by_tokens.keys()):
        for label_opt in list(grouped_by_tokens[label_set].keys()):
            # delete options that don't match num_tokens_limit
            if num_tokens_limit != -1 and label_set.num_tokens != num_tokens_limit:
                del grouped_by_tokens[label_set][label_opt]
            
            # delete incomplete label sets
            if len(grouped_by_tokens[label_set]) < original_num_options:
                del grouped_by_tokens[label_set]
                break
    
    # delete vars that are not most common token length
    if isinstance(position, int):
        assert num_tokens_limit != -1, "Must specify num-tokens when using position=each"
        for var in variables:
            if var in label_vars:
                continue
            max_count = Counter([len(tokenizer(' ' + option)["input_ids"]) for option in variables[var]]).most_common(1)[0][0]
            variables[var] = [option for option in variables[var] if len(tokenizer(' ' + option)["input_ids"]) == max_count]

    # make token options
    token_opts = list(labels.keys())
    
    # make batches
    result = []
    for batch in range(batches):
        base, src, src_labels, base_labels, pos_i, src_labels_overall, base_labels_overall = [], [], [], [], [], [], []

        # make sents
        for _ in range(batch_size):
            template = random.choice(data["templates"])
            label_opts = list(labels.keys())
            
            # pick label
            label = random.choice(label_opts)
            other_label = random.choice(label_opts)
            src_labels_overall.append(other_label)
            base_labels_overall.append(label)
            while other_label == label:
                other_label = random.choice(label_opts)
            
            # fill vars in rest of template
            base_i, src_i = fill_variables(template, variables, grouped_by_tokens, label_vars, label, other_label)
            base.append(base_i)
            src.append(src_i)

            # add labels (enforce same position in lists, assumes pairing)
            pos = random.randint(0, len(labels[label]) - 1)
            label_tok = tokenizer.encode(labels[label][pos])[0]
            other_label_tok = tokenizer.encode(labels[other_label][pos])[0]
            src_labels.append(other_label_tok)
            base_labels.append(label_tok)

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
        elif isinstance(position, int):
            pos_i = [[[position,]] * batch_size]
        else:
            raise ValueError(f"Invalid position {position}")

        # return
        result.append(Batch(pair, torch.LongTensor(src_labels), torch.LongTensor(base_labels), pos_i, src_labels_overall, base_labels_overall))
    
    return result, all_labels


def load_from_syntaxgym():
    for suite_file in glob.glob("data/test_suites/npi_ever.json"):
        print(suite_file.split('/')[-1])
        with open(suite_file, "r") as f:
            suite = json.load(f)
        if "items" not in suite:
            continue
        print(len(suite["items"]))

        region_numbers = defaultdict(set)
        for i, item in enumerate(suite["items"]):
            for condition in item["conditions"]:
                for region in condition["regions"]:
                    region_numbers[f"{condition['condition_name']}_{region['region_number']}"].add(region["content"])

        # convert sets to lists
        region_numbers = {k: str(list(v)) for k, v in region_numbers.items()}
        print(json.dumps(region_numbers, indent=2).replace("'", '"'))
        input()


def test():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.pad_token = tokenizer.eos_token
    device = "cpu"
    data, label_opts = make_data(tokenizer, "subject_verb_number_agreement_with_subject_relative_clause", 3, 2, -1, device, "label", template_file="data/templates/syntaxgym.json")
    for d in data:
        print(d[0][0])
        print(d[0][1])
        for i in range(len(d[0][0].input_ids)):
            print(tokenizer.decode(d[0][0].input_ids[i]))
            print(tokenizer.decode(d[0][1].input_ids[i]))
            print('---')
    print(data)

if __name__ == "__main__":
    print(load_from_syntaxgym())