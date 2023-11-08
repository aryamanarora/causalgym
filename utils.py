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
def get_options():
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

    return {
        "verbs": verbs,
        "names": flattened_names,
        "connectives": connectives,
    }

def make_sentence(
    tokenizer: AutoTokenizer,
    options: dict,
    name1=None,
    verb=None,
    name2=None,
    connective=None,
    token_length: int=1
):
    """Make a sentence with the given tokenizer and options"""

    # set unset vars
    while name1 is None or name1 == name2:
        test = random.choice(options['names'])
        name1 = test if len(tokenizer(test[0])['input_ids']) == token_length else name1
    while name2 is None or name1 == name2:
        test = random.choice(options['names'])
        name2 = test if len(tokenizer(' ' + test[0])['input_ids']) == token_length else name2
    while verb is None:
        test = random.choice(options['verbs'])
        verb = test if len(tokenizer(' ' + test[0])['input_ids']) == token_length else verb
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