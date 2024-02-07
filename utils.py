from torch import float32, bfloat16, float16, topk, arange
from collections import namedtuple
import random
from transformers import AutoTokenizer
import csv


# models and weight format
MODELS = [
    # "gpt2",
    # "gpt2-medium",
    # "gpt2-large",
    # "gpt2-xl",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
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
    # "gpt2": float32,
    # "gpt2-medium": float32,
    # "gpt2-large": float32,
    # "gpt2-xl": float32,
    "EleutherAI/pythia-14m": float32,
    "EleutherAI/pythia-31m": float32,
    "EleutherAI/pythia-70m": float32,
    "EleutherAI/pythia-160m": float32,
    "EleutherAI/pythia-410m": float32,
    "EleutherAI/pythia-1b": bfloat16,
    "EleutherAI/pythia-1.4b": float16,
    "EleutherAI/pythia-2.8b": float16,
    "EleutherAI/pythia-6.9b": float16,
    "EleutherAI/pythia-12b": float16,
}


parameters = {
    "pythia-12b": 11846072320,
    "pythia-6.9b": 6857302016,
    "pythia-2.8b": 2775208960,
    "pythia-1.4b": 1414647808,
    "pythia-1b": 1011781632,
    "pythia-410m": 405334016,
    "pythia-160m": 162322944,
    "pythia-70m": 70426624,
    "pythia-31m": 31000000,
    "pythia-14m": 14000000,
}


def format_token(tokenizer, tok):
    """Format the token for some path patching experiment to show decoding diff"""
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")

def top_vals(tokenizer, res, highlight=[], n=10):
    """Pretty print the top n values of a distribution over the vocabulary"""
    _, top_indices = topk(res, n)
    top_indices = top_indices.tolist() + highlight
    for i in range(len(top_indices)):
        val = top_indices[i]
        tok = format_token(tokenizer, val)
        if val in highlight:
            tok = f"\x1b[6;30;42m{tok}\x1b[0m"
            print(f"{tok:<34} {val:>5} {res[top_indices[i]].item():>10.4%}")
        else:
            print(f"{tok:<20} {val:>5} {res[top_indices[i]].item():>10.4%}")

def get_last_token(logits, attention_mask):
    last_token_indices = attention_mask.sum(1) - 1
    batch_indices = arange(logits.size(0)).unsqueeze(1)
    return logits[batch_indices, last_token_indices.unsqueeze(1)].squeeze(1)