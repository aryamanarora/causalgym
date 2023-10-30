from transformers import pipeline, set_seed
import torch
from collections import defaultdict
import argparse
import json
from tqdm import tqdm
import datetime

set_seed(42)

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


def get_bounds(text, needle):
    start = text.find(needle)
    end = start + len(needle)
    return (start, end)


@torch.no_grad()
def experiment(model="gpt2", revision="main", sequential=False, samples=100, top_k=0, top_p=1.0):
    """Run experiment."""

    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = pipeline(
        "text-generation",
        model=model,
        revision=revision,
        device=device,
        torch_dtype=torch.bfloat16 if device == "cuda:0" else torch.float32,
    )
    print("loaded model")

    # stimuli
    with open("data/stimuli.json", "r") as f:
        stimuli = json.load(f)

    # log
    log = {
        "metadata": {
            "model": model,
            "revision": revision,
            "num_parameters": generator.model.num_parameters(),
            "timestamp": str(datetime.datetime.now()),
            "samples": samples,
            "top_k": top_k,
            "top_p": top_p,
        },
        "data": {}
    }

    # generate 100 continuations
    with torch.inference_mode():
        for stimulus in stimuli:
            # get start and end positions of pronoun in text
            pronoun = get_bounds(stimulus["text"], stimulus["pronoun"])
            options = [
                get_bounds(stimulus["text"], option) for option in stimulus["options"]
            ]
            log["data"][stimulus["text"]] = {}
            log["data"][stimulus["text"]]["details"] = {"pronoun": pronoun, "options": options}

            # make sents
            sents = []
            if not sequential:
                sents = generator(
                    stimulus["text"],
                    top_k=top_k,
                    top_p=top_p,
                    max_length=50,
                    num_return_sequences=samples,
                    do_sample=True,
                )
            else:
                for _ in tqdm(range(samples)):
                    sents.append(generator(
                            stimulus["text"],
                            top_k=top_k,
                            top_p=top_p,
                            max_length=50,
                            num_return_sequences=1,
                            do_sample=True,
                    )[0])
                    torch.cuda.empty_cache()

            sents = [".".join(sent["generated_text"].split(".")[:2]) + "." for sent in sents]
            log["data"][stimulus["text"]]["sentences"] = sents

            # get entities
            counts = defaultdict(int)
            for sent in sents:
                check = sent[pronoun[1] :]
                for option in stimulus["options"]:
                    counts[option] += check.count(option)
            log["data"][stimulus["text"]]["counts"] = counts

    # dump log
    with open(f'logs/{model.replace("/", "-")}.json', "w") as f:
        json.dump(log, f, indent=4)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="name of model")
    parser.add_argument("--revision", default="main", help="revision of model")
    parser.add_argument("--sequential", action="store_true", help="run sequentially")
    parser.add_argument("--samples", default=100, type=int, help="number of samples")
    parser.add_argument("--top_k", default=0, type=int, help="top k")
    parser.add_argument("--top_p", default=1.0, type=float, help="top p")
    args = parser.parse_args()
    print(vars(args))

    if args.model == "all":
        for model in MODELS:
            args.model = model
            experiment(**vars(args))
            torch.cuda.empty_cache()
    else:
        experiment(**vars(args))


if __name__ == "__main__":
    main()
