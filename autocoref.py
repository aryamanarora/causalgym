from fastcoref import spacy_component
import spacy
import json
import glob
import os
from tqdm import tqdm

# make logs/overall directory
if not os.path.exists('logs/overall'):
    os.makedirs('logs/overall')

# load coref model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(
   "fastcoref", 
   config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
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
            second_sent_start = len(key.split('.')[0]) + 1
            res[key] = {}
            res[key]['counts'] = data[key]['counts']
            res[key]['counts_resolved'] = {option: 0 for option in data[key]['counts']}
            res[key]['counts_resolved_pronoun'] = {option: 0 for option in data[key]['counts']}

            for sent in data[key]['sentences']:

                # resolve coref
                doc = nlp(sent)
                resolved = doc._.coref_resolved
                for option in data[key]['counts']:
                    res[key]['counts_resolved'][option] += (1 if option in resolved[second_sent_start:] else 0)
                    res[key]['counts_resolved_pronoun'][option] += (1 if resolved[second_sent_start:].startswith(option) else 0)
        
        # save data
        final[file] = res

# dump final
with open('logs/overall/overall.json', 'w') as f:
    json.dump(final, f, indent=4)