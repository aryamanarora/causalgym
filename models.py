import torch

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
    # "sharpbai/alpaca-7b-merged",
    # "mistralai/Mistral-7B-v0.1",
    # "mistralai/Mistral-7B-Instruct-v0.1"
]

WEIGHTS = {
    "EleutherAI/pythia-70m": torch.bfloat16,
    "gpt2": torch.float32,
    "EleutherAI/pythia-160m": torch.bfloat16,
    "gpt2-medium": torch.float32,
    "EleutherAI/pythia-410m": torch.bfloat16,
    "gpt2-large": torch.float32,
    "EleutherAI/pythia-1b": torch.bfloat16,
    "gpt2-xl": torch.float32,
    "EleutherAI/pythia-1.4b": torch.bfloat16,
    "EleutherAI/pythia-2.8b": torch.bfloat16,
    "EleutherAI/pythia-6.9b": torch.bfloat16,
    "EleutherAI/pythia-12b": torch.bfloat16,
    "sharpbai/alpaca-7b-merged": torch.bfloat16,
    "mistralai/Mistral-7B-v0.1": torch.bfloat16,
    "mistralai/Mistral-7B-Instruct-v0.1": torch.bfloat16
}