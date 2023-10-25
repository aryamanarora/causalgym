from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
from plotnine import ggplot, aes, facet_wrap, facet_grid, geom_bar, theme, element_text, geom_errorbar, ggtitle, geom_hline, geom_point, geom_violin
from plotnine.scales import scale_color_manual, scale_x_log10, ylim
import pandas as pd
import argparse
from tqdm import tqdm
from main import MODELS

logsoftmax = torch.nn.LogSoftmax(dim=-1)

def load_data():
    # read stimuli
    with open("stimuli2.json", "r") as f:
        stimuli = json.load(f)
    names = []
    for gender in stimuli['names']:
        names.extend([(name, gender) for name in stimuli['names'][gender]])
    
    # construct all possible sentences
    sentences = {}
    for i, stimulus in enumerate(stimuli['sentences']):
        sentences[i] = []

        # replace components iteratively
        for name1 in names:
            subbed = stimulus.replace("<name1>", name1[0])
            for name2 in names:
                if name2[0] == name1[0]: continue
                genders = [name1[1], name2[1]]
                subbed2 = subbed.replace("<name2>", name2[0])
                for gender in set(genders):
                    subbed3 = subbed2.replace("<pronoun>", gender)
                    referents = []
                    if name1[1] == gender: referents.append(name1[0])
                    if name2[1] == gender: referents.append(name2[0])
                    sentences[i].append({
                        "sent": subbed3,
                        "match_name1": name1[0] in referents,
                        "match_name2": name2[0] in referents,
                        "name1": name1[0],
                        "name2": name2[0],
                        "name1_gender": name1[1],
                        "name2_gender": name2[1],
                        "pronoun_gender": gender,
                        "stimulus": i
                    })
        
        # shuffle
        random.shuffle(sentences[i])
    
    return sentences

@torch.no_grad()
def main(m: str, all_sents: list=None):
    kldivs = []
    torch.cuda.empty_cache()

    # get data
    if all_sents is None:
        all_sents = load_data()
    
    with torch.inference_mode():
        # load model
        print(m)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(m)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16).to(device)

        # generate next token distributions
        for key in all_sents:
            sentences = all_sents[key]
            distribs = []
            sents = [x['sent'] for x in sentences]
            for batch in tqdm(range(0, len(sents), 200)):
                inputs = tokenizer(sents[batch:batch+200], return_tensors="pt", padding=True).to(device)
                logits = model(**inputs).logits.to("cpu")
                probs = logsoftmax(logits)
                for i in range(probs.shape[0]):
                    distrib = probs[i, inputs['attention_mask'][i] == 1][-1]
                    distribs.append(distrib)

            # get kl divergence between distributions, picking 1000 random pairs
            for _ in tqdm(range(1000)):
                i = random.randint(0, len(sentences)-1)
                j = random.randint(0, len(sentences)-1)
                if i == j: continue
                kldiv = torch.nn.functional.kl_div(distribs[i], distribs[j], reduction="sum", log_target=True)
                label = [sentences[i]["match_name1"], sentences[i]["match_name2"], sentences[j]["match_name1"], sentences[j]["match_name2"]]
                label = "".join(["T" if x else "F" for x in label])
                kldivs.append({
                    "n11": sentences[i]["name1"],
                    "n12": sentences[i]["name2"],
                    "n21": sentences[j]["name1"],
                    "n22": sentences[j]["name2"],
                    "p1": sentences[i]["pronoun_gender"],
                    "p2": sentences[j]["pronoun_gender"],
                    "c": label,
                    "f": label[:2],
                    "s": label[2:],
                    "m": m,
                    "k": kldiv.item(),
                    "i": key
                })
    
    # dump
    with open(f"logs/kldiv/new/{m.replace('/', '-')}.json", "w") as f:
        json.dump(kldivs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="gpt2", help="name of model")
    args = parser.parse_args()
    print(vars(args))

    if args.m == "all":
        for model in MODELS:
            args.m = model
            main(**vars(args))
            torch.cuda.empty_cache()
    else:
        main(**vars(args))
