import torch
import os
import random
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from plotnine import ggplot, geom_point, aes, facet_grid, geom_line, ggtitle, geom_tile, theme, element_text, facet_wrap, geom_text
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse, scale_fill_gradient2, scale_fill_gradient
from utils import MODELS, WEIGHTS, get_last_token
from data import make_data
from eval import calculate_loss, eval, eval_sentence

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import format_token, sm, count_parameters
from models.configuration_alignable_model import (
    AlignableRepresentationConfig,
    AlignableConfig,
)
from models.alignable_base import AlignableModel
from interventions import (
    LowRankRotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    VanillaIntervention
)

def intervention_config(model_type, intervention_type, layer, num_dims, intervention_obj=None):
    if intervention_obj is None:
        intervention_obj = BoundlessRotatedSpaceIntervention if num_dims == -1 else LowRankRotatedSpaceIntervention
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
                "pos",  # intervention unit
                1,  # max number of unit
                alignable_low_rank_dimension=num_dims,  # low rank dimension
            ),
        ],
        alignable_interventions_type=intervention_obj
    )
    return alignable_config

def experiment(
    model: str,
    dataset: str,
    steps: int,
    num_dims: int,
    warmup: bool,
    eval_steps: int,
    grad_steps: int,
    batch_size: int,
    num_tokens: int,
    position: str,
    do_swap: bool,
):
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)
    print(gpt.config.num_hidden_layers)

    # train and eval sets
    trainset, labels = make_data(tokenizer, dataset, batch_size, steps, num_tokens, device, position=position)
    evalset, _ = make_data(tokenizer, dataset, 1, 20, num_tokens, device, position=position)

    # tokens to log
    tokens = tokenizer.encode("".join(labels))