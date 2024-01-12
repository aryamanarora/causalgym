from data import list_datasets
from das import experiment

def run_command(dataset):
    steps = 50
    # command = f"python das.py --model EleutherAI/pythia-70m --intervention {method} --dataset {dataset} --position each --num-tokens 1 --num-dims 1 --steps {steps}"
    experiment(
        model="EleutherAI/pythia-70m",
        dataset=dataset,
        steps=steps,
        intervention="das",
        num_dims=1,
        eval_steps=25,
        grad_steps=1,
        batch_size=4,
        num_tokens=1,
        position="each",
        intervention_site="block_output",
        store_weights=False,
    )

def main():
    # datasets = [d for d in list_datasets() if d.startswith("syntaxgym/")]
    datasets = ["gender_basic"]
    for dataset in datasets:
        run_command(dataset)

if __name__ == "__main__":
    main()