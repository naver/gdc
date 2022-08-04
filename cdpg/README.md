# Controlling Conditional Language Models without Catastrophic Forgetting

This directory contains source code accompanying the paper [Controlling Conditional Language Models without Catastrophic Forgetting](https://arxiv.org/abs/2112.00791) (ICML 2022).

## Quickstart
We assume Python 3.7+. To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

Then, you can reproduce experiments from the paper by running:
```bash
python run.py --config configs/code_pep8.json
```
substituting `code_pep8.json` with the appropriate configuration file. 

For monitoring the experiment using Weights and Biases, set `WANDB_PROJECT_NAME` and `WANDB_API_KEY`.

## Repo structure

In `configs`, we provide configuration files for the experiments described in the paper: 
* `translation_numbers_en_fr_debug.json` corresponds to the terminology-consistent translation task (section 3.2 in the paper),
* `summarization.json` corresponds to the factually-consistent summarization task (section 3.3 in the paper),
* `code_pep8.json` corresponds to the PEP8-compliant code generation task (section 3.4 in the paper),
* `code_compilability.json` corresponds to the compilable code generation task (section 3.4 in the paper).

In `contexts`, we provide contexts (e.g. source documents for summarization) we used in experiments.