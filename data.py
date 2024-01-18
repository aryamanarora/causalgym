import json
from transformers import AutoTokenizer, AutoModel
import random
import torch
from collections import defaultdict, namedtuple
import json
import glob
from typing import Union
import re
from tqdm import tqdm

random.seed(42)
Tokenized = namedtuple("Tokenized", ["base", "src", "alignment_base", "alignment_src"])


class Pair:
    """
    A pair of sentences where all features except one are held constant.

    Each pair has a base sentence and a source sentence. These two sentences
    have different "types" (the value of the differing feature) and different
    "labels" (expected continuation of the sentence).
    """

    base: list[str]
    src: list[str]
    base_type: str
    src_type: str
    base_label: str
    src_label: str


    def __init__(self, base: list[str], src: list[str], base_type: str, src_type: str, base_label: str, src_label: str):
        self.base = base
        self.src = src
        self.base_type = base_type
        self.src_type = src_type
        self.base_label = base_label
        self.src_label = src_label
    

    def tokenize(self, tokenizer: AutoTokenizer, device: str="cpu", strategy: str="suffix") -> Tokenized:
        """
        Tokenize the pair and produce alignments.

        There are many strategies for producing token-level alignments. So far,
        we try to align the suffixes or prefixes of the spans in the pair. E.g.
        given a pair like "Jeff_rey walked because" and "Emily walked because",
        we can align like [0, 2, 3] ~ [0, 1, 2] or [1, 2, 3] ~ [0, 1, 2] since
        the # of tokens between the names is different.
        """

        # produce alignments
        alignment_base, alignment_src = [], []
        pos_base, pos_src = 0, 0
        for span_i in range(len(self.base)):

            # get span lengths in tokens
            tok_base = tokenizer.tokenize(self.base[span_i])
            tok_src = tokenizer.tokenize(self.src[span_i])
            alignment = [[], []]

            # different strategies for producing alignments
            if strategy == "suffix":
                max_suffix = min(len(tok_base), len(tok_src))
                for suffix_i in range(max_suffix):
                    alignment[0].append(pos_base + len(tok_base) - max_suffix + suffix_i)
                    alignment[1].append(pos_src + len(tok_src) - max_suffix + suffix_i)
            elif strategy == "prefix":
                max_prefix = min(len(tok_base), len(tok_src))
                for prefix_i in range(max_prefix):
                    alignment[0].append(pos_base + prefix_i)
                    alignment[1].append(pos_src + prefix_i)
            
            # update positions
            alignment_base.append(alignment[0])
            alignment_src.append(alignment[1])
            pos_base += len(tok_base)
            pos_src += len(tok_src)

        # tokenize full pair and return
        base_tok = tokenizer(''.join(self.base), return_tensors="pt", padding=True).to(device)
        src_tok = tokenizer(''.join(self.src), return_tensors="pt", padding=True).to(device)
        return Tokenized(base=base_tok, src=src_tok, alignment_base=alignment_base, alignment_src=alignment_src)
    

    def swap(self) -> "Pair":
        """Swap the base and src sentences."""
        return Pair(self.src, self.base, self.src_type, self.base_type, self.src_label, self.base_label)

    
    def __repr__(self):
        return f"Pair('{self.base}' > '{self.base_label}', '{self.src}' > '{self.src_label}', {self.base_type}, {self.src_type})"


class Batch:
    """
    A Batch is a collection of Pairs that have been tokenized and padded.
    The messy part is figuring out where to do interventions, so a Batch
    encapsulates the functions for computing pos_i for the intervention
    at inference time, using the tokenized pair and alignments.
    """

    def __init__(self, pairs: list[Pair], tokenizer: AutoTokenizer, device: str="cpu", strategy: str="suffix/last"):
        self.pairs = pairs

        # tokenize base and src
        tok_strategy, pos_strategy = strategy.split('/')
        self.pos_strategy = pos_strategy
        tokenized = [pair.tokenize(tokenizer, device, tok_strategy) for pair in pairs]
        max_len = max([max(x.base.input_ids.shape[-1], x.src.input_ids.shape[-1]) for x in tokenized])
        self.base = self._stack_and_pad([x.base for x in tokenized], max_len=max_len)
        self.src = self._stack_and_pad([x.src for x in tokenized], max_len=max_len)
        self.alignment_base = [x.alignment_base for x in tokenized]
        self.alignment_src = [x.alignment_src for x in tokenized]

        # labels
        self.base_labels = torch.LongTensor([tokenizer.encode(pair.base_label)[0] for pair in pairs]).to(device)
        self.src_labels = torch.LongTensor([tokenizer.encode(pair.src_label)[0] for pair in pairs]).to(device)
        self.base_types = [pair.base_type for pair in pairs]
        self.src_types = [pair.src_type for pair in pairs]
    

    @property
    def pos(self):
        return self._compute_pos().to(self.base_labels.device)
    

    def _compute_pos(self) -> torch.LongTensor:
        """Compute pos alignments as tensors."""

        # shape of alignment: [batch_size, 2, num_spans, tokens_in_span]
        # not a proper tensor though! tokens_in_span is variable, rest is constant
        ret = []
        if self.pos_strategy == "last":
            for batch_i in range(len(self.pairs)):
                ret.append([
                    [x[-1] for x in self.alignment_src[batch_i]],
                    [x[-1] for x in self.alignment_base[batch_i]]
                ])
        elif self.pos_strategy == "first":
            for batch_i in range(len(self.pairs)):
                ret.append([
                    [x[0] for x in self.alignment_src[batch_i]],
                    [x[0] for x in self.alignment_base[batch_i]]
                ])
        
        # shape: [2, 1, batch_size, length]
        # dim 0 -> src, base (the intervention code wants src first)
        ret = torch.LongTensor(ret).permute(1, 0, 2).unsqueeze(-1)
        return ret


    def _stack_and_pad(self, input_list: dict, pad_token: int=0, max_len: int=100) -> dict:
        """Stack and pad a list of tensors outputs from a tokenizer."""
        input_ids = torch.stack([torch.nn.functional.pad(x.input_ids[0], (0, max_len - x.input_ids.shape[-1]), mode='constant', value=pad_token) for x in input_list])
        attention_mask = torch.stack([torch.nn.functional.pad(x.attention_mask[0], (0, max_len - x.attention_mask.shape[-1]), mode='constant', value=0) for x in input_list])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Dataset:
    """
    A Dataset is a template for generating minimal pairs that is loaded
    from a JSON specification.

    We importantly want examples generated from a dataset to include token-
    level alignments.
    """

    templates: list[str]
    label_vars: list[str]
    labels: dict[str, list[str]]
    variables: dict[str, Union[list[str], dict[str, list[str]]]]
    label_prepend_space: bool
    result_prepend_space: bool


    def __init__(self, data: dict):
        # load template and split it up into spans
        self.templates = data["templates"]
        self.template = re.split(r"(?<=\})|(?= \{)|(?<! )(?=\{)", '<|endoftext|>' + random.choice(self.templates))
        self.vars_per_span, self.span_names = [], []
        for token_i in range(len(self.template)):
            var = re.findall(r"\{(.+?)\}", self.template[token_i])
            self.vars_per_span.append(var)
            self.span_names.append("{" + var[0] + "}" if len(var) == 1 else self.template[token_i].replace(' ', '_'))

        # other stuff
        self.label_vars = data["label"] if isinstance(data["label"], list) else [data["label"]]
        self.labels = data["labels"]
        self.types = list(self.labels.keys())
        self.variables = data["variables"]
        self.label_prepend_space = data["label_prepend_space"]
        self.result_prepend_space = data["result_prepend_space"]

        if self.label_prepend_space:
            self.labels = {label_opt: [(" " + label) for label in self.labels[label_opt]] for label_opt in self.labels}
    

    @classmethod
    def load_from(self, template: str) -> "Dataset":
        """Load a Dataset from a json template."""
        template_file, template_name = template.split('/')
        data = json.load(open(f"data/templates/{template_file}.json", "r"))
        return Dataset(data[template_name])
    

    @property
    def length(self) -> int:
        return len(self.template)
    

    def sample_pair(self) -> Pair:
        """Sample a minimal pair from the dataset."""
        # pick types (should differ)
        base_type = random.choice(self.types)
        src_type = base_type
        while src_type == base_type:
            src_type = random.choice(self.types)

        # make template
        base, src = self.template[:], self.template[:]

        # go token by token
        for token_i in range(len(self.template)):
            var = self.vars_per_span[token_i]
            if len(var) == 0: continue
            var = var[0]
            var_temp = '{' + var + '}'

            # set label vars (different)
            if var in self.label_vars:
                base_choice = random.choice(self.variables[var][base_type])
                src_choice = random.choice(self.variables[var][src_type])
                if self.label_prepend_space:
                    base_choice = " " + base_choice
                    src_choice = " " + src_choice
                base[token_i] = base[token_i].replace(var_temp, base_choice)
                src[token_i] = src[token_i].replace(var_temp, src_choice)
            # set other vars (same for both)
            else:
                choice = random.choice(self.variables[var])
                base[token_i] = base[token_i].replace(var_temp, choice)
                src[token_i] = src[token_i].replace(var_temp, choice)
        
        # get continuations
        base_label = random.choice(self.labels[base_type])
        src_label = random.choice(self.labels[src_type])
        if self.result_prepend_space:
            base_label = " " + base_label
            src_label = " " + src_label

        return Pair(base, src, base_type, src_type, base_label, src_label)
    

    @torch.no_grad()
    def _sample_doable_pair(self, model: AutoModel, tokenizer: AutoTokenizer, device: str="cpu") -> Pair:
        """Sample a minimal pair from the dataset that is correctly labelled by a model."""
        
        # keep resampling until we get a pair that is correctly labelled
        correct = False
        while not correct:
            pair = self.sample_pair()
            base = tokenizer(''.join(pair.base), return_tensors="pt").to(device)
            src = tokenizer(''.join(pair.src), return_tensors="pt").to(device)
            base_logits = model(**base).logits[0, -1]
            src_logits = model(**src).logits[0, -1]
            base_label = tokenizer.encode(pair.base_label)[0]
            src_label = tokenizer.encode(pair.src_label)[0]
            if base_logits[base_label] > base_logits[src_label] and src_logits[src_label] > src_logits[base_label]:
                correct = True
            
        return pair


    def sample_batch(
            self, tokenizer: AutoTokenizer, batch_size: int, device: str="cpu",
            strategy="suffix/last", model: Union[AutoModel, None]=None) -> Batch:
        """Sample a batch of minimal pairs from the dataset."""
        pairs = [
            self.sample_pair()
            if model is None else
            self._sample_doable_pair(model, tokenizer, device)
            for _ in range(batch_size // 2)
        ]
        for i in range(batch_size // 2):
            pairs.append(pairs[i].swap())
        return Batch(pairs, tokenizer, device, strategy)


    def sample_batches(
            self, tokenizer: AutoTokenizer, batch_size: int, num_batches: int,
            device: str="cpu", strategy="suffix/last", seed: int=42,
            model: Union[AutoModel, None]=None) -> list[Batch]:
        """Sample a list of batches of minimal pairs from the dataset."""
        random.seed(seed)
        return [self.sample_batch(tokenizer, batch_size, device, strategy, model) for _ in tqdm(range(num_batches))]


def load_from_syntaxgym():
    for suite_file in glob.glob("data/test_suites/subordination.json"):
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


if __name__ == "__main__":
    print(load_from_syntaxgym())