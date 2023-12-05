# This library is our indicator that the required installs
# need to be done.
import transformers
import sys
sys.path.append("/juice2/scr2/aryaman/align-transformers/")
print(sys.path)

import sys
import torch
sys.path.append("..")

import pandas as pd
from models.utils import embed_to_distrib, top_vals, format_token, sm
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention
from models.gpt_neox.modelings_alignable_gpt_neox import create_gpt_neox

# %config InlineBackend.figure_formats = ['svg']
from plotnine import ggplot, geom_tile, aes, facet_wrap, theme, element_text, \
                     geom_bar, geom_hline, scale_y_log10, geom_point, geom_label, facet_grid, \
                     geom_line, geom_area
from plotnine.scales import scale_y_reverse, scale_fill_cmap, scale_y_continuous, scale_color_cmap
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = "EleutherAI/pythia-1.4b"
config, tokenizer, gpt = create_gpt_neox(name=model, cache_dir="/nlp/scr/aryaman/.cache/huggingface/hub")
gpt.to(device)

def simple_position_config(model_type, intervention_type="block_output", layer=0):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=VanillaIntervention,
    )
    return alignable_config

def attn_outputs_config(model_type, layer):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                "attention_output", # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            )
        ],
        alignable_interventions_type=VanillaIntervention,
    )
    return alignable_config

def attn_heads_config(model_type, layer):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                "head_attention_value_output", # intervention type
                "h",             # intervention unit
                1                  # max number of unit
            ),
            AlignableRepresentationConfig(
                layer,             # layer
                "attention_output", # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            )
        ],
        alignable_interventions_type=VanillaIntervention,
    )
    return alignable_config

data = []

# tokens = [tokenizer.encode(x)[0] for x in [" Sarah", " Elizabeth"]]
# base = tokenizer(f"Sarah (Naples, 21 December 1586 (date of baptism)[1] – Naples, c. 1642) was an Italian composer and singer. She was born and died in Naples. From 1610,", return_tensors="pt").to(device)
# # sources = [tokenizer(f"John (Naples, 21 December 1586 (date of baptism)[1] – Naples, c. 1642) was an Italian composer and singer. She was born and died in Naples. From 1610,", return_tensors="pt").to(device)]
# sources = [tokenizer(f"Elizabeth (Naples, 21 December 1586 (date of baptism)[1] – Naples, c. 1642) was an Italian composer and singer. She was born and died in Naples. From 1610,", return_tensors="pt").to(device)]
# top_vals(tokenizer, sm(gpt(**base).logits)[0, -1], 10)
# print('---')
# top_vals(tokenizer, sm(gpt(**sources[0]).logits)[0, -1], 10)

tokens = tokenizer.encode(f" he she")
base = tokenizer(f"Bill suddenly amazed John because", return_tensors="pt").to(device)
sources = [tokenizer(f"Sarah suddenly amazed John because", return_tensors="pt").to(device)]
base_logits = gpt(**base).logits[0, -1].cpu().detach()
base_probs = sm(gpt(**base).logits)[0, -1].cpu().detach()
# tokens = tokenizer.encode(f" France Italy")
# base = tokenizer(f"French is the official language of", return_tensors="pt").to(device)
# sources = [tokenizer(f"Italian is the official language of", return_tensors="pt").to(device)]

assert base.input_ids.shape == sources[0].input_ids.shape

length = base.input_ids.shape[-1]
for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
    for head_i in range(config.num_attention_heads):
        # for intervention_type in ["block_output"]:
        for pos_i in range(length):  
            alignable_config = attn_heads_config(type(gpt), layer_i)
            alignable = AlignableModel(alignable_config, gpt)
            _, counterfactual_outputs = alignable(
                base,
                [sources[0], base],
                {"sources->base": (
                    [[[head_i for x in range(length - 1)]], [[x for x in range(length) if x != pos_i]]],
                    [[[head_i for x in range(length - 1)]], [[x for x in range(length) if x != pos_i]]]
                )}
            )
            probs = sm(counterfactual_outputs.logits)[0, -1].cpu().detach()
            logits = counterfactual_outputs.logits[0, -1].cpu().detach()
            for i, token in enumerate(tokens):
                print(f"layer {layer_i}, head {head_i}, pos {pos_i}: {probs[token].item()} ({format_token(tokenizer, token)})")
                data.append({
                    "layer": layer_i,
                    "pos": pos_i,
                    "head": head_i,
                    "prob": probs[token].item(),
                    "probdiff": probs[token].item() - base_probs[token].item(),
                    "logit": logits[token].item(),
                    "logitdiff": logits[token].item() - base_logits[token].item(),
                    "token": format_token(tokenizer, token),
                    # "type": intervention_type
                })

df = pd.DataFrame(data)
df["layer"] = df["layer"].astype(float)
df["pos"] = df["pos"].astype(int)
df["prob"] = df["prob"].astype(float)
df["logit"] = df["logit"].astype(float)
df["logitdiff"] = df["logitdiff"].astype(float)

# print top probs
print(df[df["token"] == "_she"].sort_values("prob", ascending=False).head(10))

# Group by 'pos' and sum the specified columns
# grouped = df.groupby(['layer', 'type', 'token'])[['prob', 'logit', 'logitdiff']].sum().reset_index()
# grouped['pos'] = -1
# grouped = grouped[['layer', 'pos', 'prob', 'logit', 'logitdiff', 'token', 'type']]
# df = pd.concat([df, grouped], ignore_index=True)

labels = [format_token(tokenizer, x) if sources[0].input_ids[0, i] == x else f"{format_token(tokenizer, x)} / {format_token(tokenizer, sources[0].input_ids[0, i])}" for i, x in enumerate(base.input_ids[0])][::-1]
plot = (ggplot(df, aes(x="layer", y="pos"))
        + scale_y_continuous(breaks=[-1] + list(range(base.input_ids.shape[-1]))[::-1], labels=['ALL'] + labels)
        + geom_tile(aes(fill="prob")) + scale_fill_cmap("Purples") + facet_grid("type ~ token"))
# plot = (ggplot(df, aes(x="layer", y="logitdiff")) + facet_grid("pos ~ token") + geom_area())
plot.save(f"figs/bob2.pdf")