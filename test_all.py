import os
from data import list_datasets

def run_command(dataset, method):
    steps = 125
    command = f"python das.py --model EleutherAI/pythia-70m --intervention {method} --dataset {dataset} --position each --num-tokens 1 --num-dims 1 --steps {steps}"
    os.system(command)

def main():
    methods = ["das", "probe_sklearn"]
    datasets = [d for d in list_datasets() if d.startswith("syntaxgym/")]
    for dataset in datasets:
        for method in methods:
            run_command(dataset, method)

if __name__ == "__main__":
    main()