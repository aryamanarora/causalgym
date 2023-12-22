from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import WEIGHTS, top_vals, format_token
import torch

with torch.no_grad():
    # load model
    model = input("Model: ")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)

    # make data
    while True:
        text = input("Text: ")
        text = tokenizer(text, return_tensors="pt").to(device)
        logits = gpt(**text).logits[0, -1]
        probs = logits.softmax(-1)
        top_vals(tokenizer, probs)