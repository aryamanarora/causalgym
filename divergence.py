from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
from plotnine import ggplot, aes, facet_wrap, facet_grid, geom_bar, theme, element_text, geom_errorbar, ggtitle, geom_hline, geom_point, geom_violin
from plotnine.scales import scale_color_manual, scale_x_log10, ylim
import pandas as pd
import argparse
from tqdm import tqdm
from main import MODELS, WEIGHTS

logsoftmax = torch.nn.LogSoftmax(dim=-1)
softmax = torch.nn.Softmax(dim=-1)

def load_data(referent="pronoun"):
    # read stimuli
    with open("data/stimuli2.json", "r") as f:
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
                if referent == "pronoun":
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
                elif referent == "name":
                    for name in [name1[0], name2[0]]:
                        subbed3 = subbed2.replace("<pronoun>", name)
                        sentences[i].append({
                            "sent": subbed3,
                            "match_name1": name1[0] == name,
                            "match_name2": name2[0] == name,
                            "name1": name1[0],
                            "name2": name2[0],
                            "name1_gender": name1[1],
                            "name2_gender": name2[1],
                            "pronoun_gender": name,
                            "stimulus": i
                        })
        
        # shuffle
        random.shuffle(sentences[i])
    
    return sentences

@torch.no_grad()
def main(
    m: str,
    metric_name: str="js_div",
    use_names=False
):
    results = []
    torch.cuda.empty_cache()

    # get data
    all_sents = load_data()
    all_sents2 = None
    if use_names:
        all_sents2 = load_data(referent="name")
    
    with torch.inference_mode():
        # load model
        print(m)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(m)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            m,
            torch_dtype=WEIGHTS.get(m, torch.bfloat16) if device == "cuda:0" else torch.float32
        ).to(device)

        # generate next token distributions
        for key in tqdm(all_sents):

            # get data for this set of templates
            sentences = all_sents[key]
            sentences2 = None
            if use_names:
                sentences2 = all_sents2[key]
            distribs, distribs2 = [], []

            # inference, get logits and smooth them with epsilon (avoid NaNs in metric)
            sents = [x['sent'] for x in sentences]
            for batch in range(0, len(sents), 200):
                inputs = tokenizer(sents[batch:batch+200], return_tensors="pt", padding=True).to(device)
                out = model(**inputs)
                logits = out.logits.to("cpu")
                probs = softmax(logits)
                for i in range(probs.shape[0]):
                    distrib = probs[i, inputs['attention_mask'][i] == 1][-1]
                    distrib += 1e-9
                    distribs.append(distrib)

            # names separately
            if use_names:
                sents2 = [x['sent'] for x in sentences2]
                for batch in range(0, len(sents2), 200):
                    inputs = tokenizer(sents2[batch:batch+200], return_tensors="pt", padding=True).to(device)
                    out = model(**inputs)
                    logits = out.logits.to("cpu")
                    probs = softmax(logits)
                    for i in range(probs.shape[0]):
                        distrib = probs[i, inputs['attention_mask'][i] == 1][-1]
                        distrib += 1e-9
                        distribs2.append(distrib)

            # get kl divergence between distributions, picking 1000 random pairs
            for _ in range(1000):
                i = random.randint(0, len(sentences)-1)
                j = random.randint(0, len(sentences)-1) if not use_names else random.randint(0, len(sentences)-1)
                if i == j: continue

                # compute metrics
                cur = [distribs[i], distribs[j]] if not use_names else [distribs[i], distribs2[j]]
                
                metric = None
                if metric_name == "kl_div":
                    metric = torch.nn.functional.kl_div(cur[0].log(), cur[1].log(), reduction="sum", log_target=True)
                elif metric_name == "js_div":
                    mixture = (cur[0] + cur[1]) / 2
                    metric = (torch.nn.functional.kl_div(cur[0].log(), mixture.log(), reduction="sum", log_target=True) + torch.nn.functional.kl_div(cur[1].log(), mixture.log(), reduction="sum", log_target=True)) / 2.0
                elif metric_name == "cosine":
                    metric = torch.nn.functional.cosine_similarity(cur[0].unsqueeze(0), cur[1].unsqueeze(0))

                first = sentences[i]
                second = sentences[j] if not use_names else sentences2[j]
                
                label = [first["match_name1"], first["match_name2"], second["match_name1"], second["match_name2"]]
                label = "".join(["T" if x else "F" for x in label])
                
                results.append({
                    "n11": first["name1"],
                    "n12": first["name2"],
                    "n21": second["name1"],
                    "n22": second["name2"],
                    "p1": first["pronoun_gender"],
                    "p2": second["pronoun_gender"],
                    "c": label,
                    "f": label[:2],
                    "s": label[2:],
                    "m": m,
                    "k": metric.item(),
                    "i": key
                })
    
    # dump
    with open(f"logs/kldiv/new/{m.replace('/', '-')}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="gpt2", help="name of model")
    parser.add_argument("--metric_name", default="kl_div", help="metric to use")
    parser.add_argument("--use_names", action="store_true", help="compare pronouns with names")
    args = parser.parse_args()
    print(vars(args))
    
    if args.m == "all":
        for model in tqdm(MODELS):
            args.m = model
            main(**vars(args))
            torch.cuda.empty_cache()
    else:
        main(**vars(args))
