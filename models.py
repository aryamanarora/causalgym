import torch

MODELS = [
    "EleutherAI/pythia-70m",
    "gpt2",
    "EleutherAI/pythia-160m",
    "gpt2-medium",
    "EleutherAI/pythia-410m",
    "gpt2-large",
    "EleutherAI/pythia-1b",
    "gpt2-xl",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "sharpbai/alpaca-7b-merged",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1"
]

WEIGHTS = {
    "EleutherAI/pythia-70m": torch.bfloat16,
    "gpt2": torch.float16,
    "EleutherAI/pythia-160m": torch.bfloat16,
    "gpt2-medium": torch.float16,
    "EleutherAI/pythia-410m": torch.bfloat16,
    "gpt2-large": torch.float16,
    "EleutherAI/pythia-1b": torch.bfloat16,
    "gpt2-xl": torch.float16,
    "EleutherAI/pythia-1.4b": torch.bfloat16,
    "EleutherAI/pythia-2.8b": torch.bfloat16,
    "sharpbai/alpaca-7b-merged": torch.bfloat16,
    "mistralai/Mistral-7B-v0.1": torch.bfloat16,
    "mistralai/Mistral-7B-Instruct-v0.1": torch.bfloat16
}