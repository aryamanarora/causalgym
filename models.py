from torch import float32, bfloat16

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
    "EleutherAI/pythia-70m": float32,
    "gpt2": float32,
    "EleutherAI/pythia-160m": float32,
    "gpt2-medium": float32,
    "EleutherAI/pythia-410m": float32,
    "gpt2-large": float32,
    "EleutherAI/pythia-1b": bfloat16,
    "gpt2-xl": float32,
    "EleutherAI/pythia-1.4b": float32,
    "EleutherAI/pythia-2.8b": float32,
    "EleutherAI/pythia-6.9b": bfloat16,
    "EleutherAI/pythia-12b": bfloat16,
    "sharpbai/alpaca-7b-merged": bfloat16,
    "mistralai/Mistral-7B-v0.1": bfloat16,
    "mistralai/Mistral-7B-Instruct-v0.1": bfloat16
}