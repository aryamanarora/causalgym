from fastcoref import spacy_component
import spacy
import json
import glob
import os
from tqdm import tqdm
import torch
import pandas as pd
from plotnine import ggplot, aes, facet_wrap, facet_grid, geom_bar, theme, element_text
import argparse

ORDER = ['EleutherAI-pythia-70m', 'gpt2', 'EleutherAI-pythia-160m', 'gpt2-medium', 'EleutherAI-pythia-410m', 'gpt2-large', 'EleutherAI-pythia-1b', 'gpt2-xl', 'EleutherAI-pythia-1.4b', 'EleutherAI-pythia-2.8b', 'sharpbai-alpaca-7b-merged']

def autocoref():
    # make logs/overall directory
    if not os.path.exists('logs/overall'):
        os.makedirs('logs/overall')

    # load coref model
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
    "fastcoref", 
    config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
    )

    final = {}

    # for each file
    for file in tqdm(glob.glob("logs/*.json")):
        with open(file, 'r') as f:

            # load data
            data = json.load(f)
            res = {}

            # run coref
            for key in data:
                res[key] = {}
                res[key]['counts'] = data[key]['counts']
                res[key]['counts_resolved'] = {option: 0 for option in data[key]['counts']}
                res[key]['counts_resolved_pronoun'] = {option: 0 for option in data[key]['counts']}

                for sent in data[key]['sentences']:
                    # resolve coref
                    doc = nlp(
                        sent,
                        component_cfg={"fastcoref": {'resolve_text': True}}
                    )
                    resolved = doc._.resolved_text
                    for option in data[key]['counts']:
                        res[key]['counts_resolved'][option] += (1 if option in '.'.join(resolved.split('.')[1:]) else 0)
                        res[key]['counts_resolved_pronoun'][option] += (1 if resolved.split('. ')[1].startswith(option) else 0)
            
            # save data
            final[file] = res

    # dump final
    with open('logs/overall/overall.json', 'w') as f:
        json.dump(final, f, indent=4)

def plot():
    """Plot the results of autocoref"""

    with open('logs/overall/overall.json', 'r') as f:
        data = json.load(f)
    
    # prepare pandas
    rows = []
    for key in data:
        model = key[len('logs/'):-len('.json')]
        for sent in data[key]:
            for metric in data[key][sent]:
                for option in data[key][sent][metric]:
                    rows.append({
                        "model": model,
                        "sent": sent,
                        "metric": metric,
                        "option": option,
                        "count": data[key][sent][metric][option]
                    })
    
    # df, set model order
    df = pd.DataFrame(rows)
    df['model'] = pd.Categorical(df['model'].astype(str))
    df['model'].cat.set_categories(ORDER, inplace=True)

    # plot
    plot = (ggplot(df, aes(x="model", y="count", fill="option"))
            + geom_bar(stat="identity") + facet_grid("metric~sent")
            + theme(figure_size=(10, 6), axis_text_x=element_text(rotation=45, hjust=1)))
    plot.save("logs/overall/plot.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autocoref", action="store_true", help="run autocoref")
    parser.add_argument("--plot", action="store_true", help="plot autocoref results")
    args = parser.parse_args()

    if args.autocoref:
        autocoref()
    if args.plot:
        plot()

if __name__ == "__main__":
    main()