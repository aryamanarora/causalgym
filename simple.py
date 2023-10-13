from transformers import pipeline, set_seed
import torch
from collections import defaultdict
import argparse
import json
import os
from tqdm import tqdm
import datetime

set_seed(42)

MODELS = ['EleutherAI/pythia-70m', 'gpt2', 'EleutherAI/pythia-160m', 'gpt2-medium', 'EleutherAI/pythia-410m', 'gpt2-large', 'EleutherAI/pythia-1b', 'gpt2-xl', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b', 'sharpbai/alpaca-7b-merged']

def get_bounds(text, needle):
    start = text.find(needle)
    end = start + len(needle)
    return (start, end)

@torch.no_grad()
def experiment(model="gpt2", revision="main", use_local_cache=False, sequential=False):
    # load model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ['TRANSFORMERS_CACHE'] = '../.huggingface_cache' if use_local_cache else '~/.cache/huggingface/hub'
    generator = pipeline('text-generation', model=model, revision=revision, device=device, torch_dtype=torch.bfloat16 if device == 'cuda:0' else torch.float32)
    print("loaded model")

    # stimuli
    stimuli = [
        {"text": "John seized the comic from Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
        {"text": "John passed the comic to Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
        {"text": "John passed a comic to Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
        {"text": "John was passing a comic to Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
    ]

    # log
    log = {'metadata': {
        'model': model,
        'revision': revision,
        'num_parameters': generator.model.num_parameters(),
        'timestamp': str(datetime.datetime.now())
    }}

    # generate 100 continuations
    with torch.inference_mode():
        for stimulus in stimuli:
            # get start and end positions of pronoun in text
            pronoun = get_bounds(stimulus['text'], stimulus['pronoun'])
            options = [get_bounds(stimulus['text'], option) for option in stimulus['options']]
            log[stimulus['text']] = {}
            log[stimulus['text']]['details'] = {
                'pronoun': pronoun,
                'options': options
            }

            # make sents
            sents = []
            if not sequential:
                sents = generator(stimulus['text'], max_length=50, num_return_sequences=100, do_sample=True)
            else:
                for _ in tqdm(range(100)):
                    sents.append(generator(stimulus['text'], max_length=50, num_return_sequences=1, do_sample=True)[0])
                    torch.cuda.empty_cache()
            sents = ['.'.join(sent['generated_text'].split('.')[:2]) + '.' for sent in sents]
            log[stimulus['text']]['sentences'] = sents

            # get entities
            counts = defaultdict(int)
            for sent in sents:
                check = sent[pronoun[1]:]
                for option in stimulus['options']:
                    counts[option] += check.count(option)
            log[stimulus['text']]['counts'] = counts

    # dump log
    with open(f'logs/{model.replace("/", "-")}.json', 'w') as f:
        json.dump(log, f, indent=4)
    
    return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', help='name of model')
    parser.add_argument('--revision', default='main', help='revision of model')
    parser.add_argument('--use-local-cache', action='store_true', help='use user cache on cluster')
    parser.add_argument('--sequential', action='store_true', help='run sequentially')
    args = parser.parse_args()

    if args.model == 'all':
        for model in MODELS:
            args.model = model
            experiment(**vars(args))
            torch.cuda.empty_cache()
    else:
        experiment(**vars(args))

if __name__ == '__main__':
    main()
