from torch import float32, bfloat16, float16
from collections import namedtuple
import random
from transformers import AutoTokenizer
import csv

# models and weight format
MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

WEIGHTS = {
    "gpt2": float32,
    "gpt2-medium": float32,
    "gpt2-large": float32,
    "gpt2-xl": float32,
    "EleutherAI/pythia-70m": float32,
    "EleutherAI/pythia-160m": float32,
    "EleutherAI/pythia-410m": float32,
    "EleutherAI/pythia-1b": bfloat16,
    "EleutherAI/pythia-1.4b": float32,
    "EleutherAI/pythia-2.8b": float32,
    "EleutherAI/pythia-6.9b": float16,
    "EleutherAI/pythia-12b": float16,
}

# sentences
Sentence = namedtuple("Sentence", ["sentence", "verb", "name1", "name2", "connective"])

# data things
def get_options(tokenizer: AutoTokenizer=None, token_length: int=None):
    """Get options for experiment."""

    # verbs
    with open('data/ferstl.csv', 'r') as f:
        reader = csv.reader(f)
        verbs = [tuple(x) for x in reader]
    
    # names
    names = {
        "he": ["John", "Bill", "Joseph", "Patrick", "Ken", "Geoff", "Simon", "Richard", "David", "Michael"],
        "she": ["Amanda", "Britney", "Catherine", "Dorothy", "Elizabeth", "Fiona", "Gina", "Helen", "Irene", "Jane"]
    }
    flattened_names = [(name, gender) for gender in names for name in names[gender]]

    # connectives
    connectives = ["because"]

    # filter
    if token_length is not None:
        verbs = [x for x in verbs if len(tokenizer(' ' + x[0])['input_ids']) == token_length]
        flattened_names = [x for x in flattened_names if len(tokenizer(x[0])['input_ids']) == token_length]
        connectives = [x for x in connectives if len(tokenizer(' ' + x)['input_ids']) == token_length]

    return {
        "verbs": verbs,
        "names": flattened_names,
        "connectives": connectives,
    }

def make_sentence(
    options: dict,
    name1=None,
    verb=None,
    name2=None,
    connective=None
):
    """Make a sentence with the given tokenizer and options"""

    # set unset vars
    while name1 is None or name1 == name2:
        name1 = random.choice(options['names'])
    while name2 is None or name1 == name2:
        name2 = random.choice(options['names'])
    while verb is None:
        verb = random.choice(options['verbs'])
    while connective is None:
        connective = random.choice(options['connectives'])

    # make sentence
    sent = f"{name1[0]} {verb[0]} {name2[0]} {connective}"
    return Sentence(
        sentence=sent,
        verb=verb,
        name1=name1,
        name2=name2,
        connective=connective,
    )