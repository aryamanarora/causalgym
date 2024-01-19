from data import list_datasets
from das import experiment

def run_command(dataset):
    # command = f"python das.py --model EleutherAI/pythia-70m --intervention {method} --dataset {dataset} --position each --num-tokens 1 --num-dims 1 --steps {steps}"
    print(dataset)
    experiment(
        model="EleutherAI/pythia-70m",
        dataset=dataset,
        steps=200,
        eval_steps=25,
        grad_steps=1,
        batch_size=4,
        intervention_site="block_output",
    )

def main():
    datasets = [d for d in list_datasets() if d.startswith("syntaxgym/")]
    for dataset in datasets:
        run_command(dataset)

if __name__ == "__main__":
    main()