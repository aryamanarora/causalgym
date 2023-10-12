from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed
import torch
from collections import defaultdict
import argparse
import json

set_seed(42)

def get_bounds(text, needle):
    start = text.find(needle)
    end = start + len(needle)
    return (start, end)

def experiment(model="gpt2", revision="main", use_local_cache=False):
    # load model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cache_dir = '/nlp/scr/aryaman/.huggingface_cache' if use_local_cache else None
    if cache_dir:
        generator = pipeline('text-generation', model=model, revision=revision, device=device, cache_dir=cache_dir)
    else:
        generator = pipeline('text-generation', model=model, revision=revision, device=device)
    print("loaded model")

    # stimuli
    stimuli = [
        {"text": "John seized the comic from Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
        {"text": "John passed the comic to Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
    ]

    # log
    log = {}

    # generate 100 continuations
    with torch.inference_mode():
        for stimulus in stimuli:
            # get start and end positions of pronoun in text
            pronoun = get_bounds(stimulus['text'], stimulus['pronoun'])
            options = [get_bounds(stimulus['text'], option) for option in stimulus['options']]
            log[stimulus['text']] = {}

            # make sents
            sents = generator(stimulus['text'], max_length=50, num_return_sequences=100)
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
    with open(f'logs/{model}.json', 'w') as f:
        json.dump(log, f, indent=4)
    
    return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', help='name of model')
    parser.add_argument('--revision', default='main', help='revision of model')
    parser.add_argument('--use-local-cache', action='store_true', help='use user cache on cluster')
    args = parser.parse_args()
    experiment(**vars(args))

if __name__ == '__main__':
    main()