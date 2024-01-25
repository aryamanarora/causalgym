from data import list_datasets
from das import experiment
import argparse

def run_command(model: str, dataset: str):
    # command = f"python das.py --model EleutherAI/pythia-70m --intervention {method} --dataset {dataset} --position each --num-tokens 1 --num-dims 1 --steps {steps}"
    print(dataset)
    experiment(
        model=model,
        dataset=dataset,
        steps=100,
        eval_steps=25,
        grad_steps=1,
        batch_size=4,
        intervention_site="block_output",
        strategy="last",
        lr=5e-3,
        only_das=False,
    )

def main(model: str):
    datasets = [d for d in list_datasets() if d.startswith("syntaxgym/")]
    for dataset in datasets:
        run_command(model, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    args = parser.parse_args()
    main(args.model)