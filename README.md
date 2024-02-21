# CausalGym

*Paper*: **[Arora et al. (2024)](https://arxiv.org/abs/2402.12560)**

**CausalGym** is a benchmark for comparing the performance of causal interpretability methods on a variety of simple linguistic tasks taken from the SyntaxGym evaluation set ([Gauthier et al., 2020](https://aclanthology.org/2020.acl-demos.10/), [Hu et al., 2020](https://aclanthology.org/2020.acl-main.158/)) and converted into a format suitable for interventional interpretability.

This repository includes code for:
- Training DAS and all the other methods benchmarked in the paper, on every region, layer, and task for some model. This is sufficient for replicating all experiments in the paper (including hyperparameter sweeps and interpretability during training).
- Reproducing every plot in the paper.
- Template specifications for every task in the benchmark and utils for generating examples, tokenizing, generating non-overlapping train/test sets, and so on.
- Testing model outputs on the task templates; this was used to design the benchmark tasks.

You can also download the train/dev/test splits for each task as used in the paper via [HuggingFace](https://huggingface.co/datasets/aryaman/causalgym).

If you are having trouble getting anything running, do not hesitate to file an issue! We would love to help you benchmark your new method or help you replicate the results from our paper.

## Instructions

> [!IMPORTANT]  
> The implementations in this repo are only for `GPTNeoX`-type language models (e.g. the `pythia` series) and will probably not work for other architectures without some modifications.

First install the requirements (a fresh environment is probably best):

```bash
pip install -r requirements.txt
```

To train every method, layer, region, and task for `pythia-70m` (results are logged to the directory `logs/das/`):

```bash
python test_all.py --model EleutherAI/pythia-70m
```

To do the same but with the dog-give control task used to compute selectivity:

```bash
python test_all.py --model EleutherAI/pythia-70m --manipulate dog-give
```

Once you have run this for several models, you can create results tables with:

```bash
python plot.py --file logs/das/ --plot summary --metrics odds --reload
```

This caches intermediate results in csv file in the directory. To produce the causal tracing-style plots:

```bash
python plot.py --file logs/das/ --plot pos_all --metrics odds
```

## Citation

Please cite the CausalGym preprint:

```bibtex
@article{arora-etal-2024-causalgym,
    title = "{C}ausal{G}ym: Benchmarking causal interpretability methods on linguistic tasks",
    author = "Arora, Aryaman and Jurafsky, Dan and Potts, Christopher",
    journal = "arXiv:2402.12560",
    year = "2024",
    url = "https://arxiv.org/abs/2402.12560"
}
```

Also cite the earlier SyntaxGym papers:

```bibtex
@inproceedings{gauthier-etal-2020-syntaxgym,
    title = "{S}yntax{G}ym: An Online Platform for Targeted Evaluation of Language Models",
    author = "Gauthier, Jon and Hu, Jennifer and Wilcox, Ethan and Qian, Peng and Levy, Roger",
    editor = "Celikyilmaz, Asli and Wen, Tsung-Hsien",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-demos.10",
    doi = "10.18653/v1/2020.acl-demos.10",
    pages = "70--76",
}

@inproceedings{hu-etal-2020-systematic,
    title = "A Systematic Assessment of Syntactic Generalization in Neural Language Models",
    author = "Hu, Jennifer and Gauthier, Jon and Qian, Peng and Wilcox, Ethan and Levy, Roger",
    editor = "Jurafsky, Dan and Chai, Joyce and Schluter, Natalie and Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.158",
    doi = "10.18653/v1/2020.acl-main.158",
    pages = "1725--1744",
}
```

## Task examples

| **Task**                             | **Example**                                                                                                                                                 |
|:-------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ***Agreement*** (4)                  |                                                                                                                                                             |
| `agr_gender`                         | \[**John**\]\[**Jane**\] walked because \[**he**\]\[**she**\]                                                                                               |
| `agr_sv_num_subj-relc`               | The \[**guard**\]\[**guards**\] that hated the manager \[**is**\]\[**are**\]                                                                                |
| `agr_sv_num_obj-relc`                | The \[**guard**\]\[**guards**\] that the customers hated \[**is**\]\[**are**\]                                                                              |
| `agr_sv_num_pp`                      | The \[**guard**\]\[**guards**\] behind the managers \[**is**\]\[**are**\]                                                                                   |
| ***Licensing*** (7)                  |                                                                                                                                                             |
| `agr_refl_num_subj-relc`             | The \[**farmer**\]\[**farmers**\] that loved the actors embarrassed \[**himself**\]\[**themselves**\]                                                       |
| `agr_refl_num_obj-relc`              | The \[**farmer**\]\[**farmers**\] that the actors loved embarrassed \[**himself**\]\[**themselves**\]                                                       |
| `agr_refl_num_pp`                    | The \[**farmer**\]\[**farmers**\] behind the actors embarrassed \[**himself**\]\[**themselves**\]                                                           |
| `npi_any_subj-relc`                  | \[**No**\]\[**The**\] consultant that has helped the taxi driver has shown \[**any**\]\[**some**\]                                                          |
| `npi_any_obj-relc`                   | \[**No**\]\[**The**\] consultant that the taxi driver has helped has shown \[**any**\]\[**some**\]                                                          |
| `npi_ever_subj-relc`                 | \[**No**\]\[**The**\] consultant that has helped the taxi driver has \[**ever**\]\[**never**\]                                                              |
| `npi_ever_obj-relc`                  | \[**No**\]\[**The**\] consultant that the taxi driver has helped has \[**ever**\]\[**never**\]                                                              |
| ***Garden path effects*** (6)        |                                                                                                                                                             |
| `garden_mvrr`                        | The infant \[**who was**\]\[**⌀**\] brought the sandwich from the kitchen \[**by**\]\[**.**\]                                                               |
| `garden_mvrr_mod`                    | The infant \[**who was**\]\[**⌀**\] brought the sandwich from the kitchen with a new microwave \[**by**\]\[**.**\]                                          |
| `garden_npz_obj`                     | While the students dressed \[**,**\]\[**⌀**\] the comedian \[**was**\]\[**for**\]                                                                           |
| `garden_npz_obj_mod`                 | While the students dressed \[**,**\]\[**⌀**\] the comedian who told bad jokes \[**was**\]\[**for**\]                                                        |
| `garden_npz_v-trans`                 | As the criminal \[**slept**\]\[**shot**\] the woman \[**was**\]\[**for**\]                                                                                  |
| `garden_npz_v-trans_mod`             | As the criminal \[**slept**\]\[**shot**\] the woman who told bad jokes \[**was**\]\[**for**\]                                                               |
| ***Gross syntactic state*** (4)      |                                                                                                                                                             |
| `gss_subord`                         | \[**While the**\]\[**The**\] lawyers lost the plans \[**they**\]\[**.**\]                                                                                   |
| `gss_subord_subj-relc`               | \[**While the**\]\[**The**\] lawyers who wore white lab jackets studied the book that described several advances in cancer therapy \[**,**\]\[**.**\]       |
| `gss_subord_obj-relc`                | \[**While the**\]\[**The**\] lawyers who the spy had contacted repeatedly studied the book that colleagues had written on cancer therapy \[**,**\]\[**.**\] |
| `gss_subord_pp`                      | \[**While the**\]\[**The**\] lawyers in a long white lab jacket studied the book about several recent advances in cancer therapy \[**,**\]\[**.**\]         |
| ***Long-distance dependencies*** (8) |                                                                                                                                                             |
| `cleft`                              | What the young man \[**did**\]\[**ate**\] was \[**make**\]\[**for**\]                                                                                       |
| `cleft_mod`                          | What the young man \[**did**\]\[**ate**\] after the ingredients had been bought from the store was \[**make**\]\[**for**\]                                  |
| `filler_gap_embed_3`                 | I know \[**that**\]\[**what**\] the mother said the friend remarked the park attendant reported your friend sent \[**him**\]\[**.**\]                       |
| `filler_gap_embed_4`                 | I know \[**that**\]\[**what**\] the mother said the friend remarked the park attendant reported the cop thinks your friend sent \[**him**\]\[**.**\]        |
| `filler_gap_hierarchy`               | The fact that the brother said \[**that**\]\[**who**\] the friend trusted \[**the**\]\[**was**\]                                                            |
| `filler_gap_obj`                     | I know \[**that**\]\[**what**\] the uncle grabbed \[**him**\]\[**.**\]                                                                                      |
| `filler_gap_pp`                      | I know \[**that**\]\[**what**\] the uncle grabbed food in front of \[**him**\]\[**.**\]                                                                     |
| `filler_gap_subj`                    | I know \[**that**\]\[**who**\] the uncle grabbed food in front of \[**him**\]\[**.**\]                                                                      |