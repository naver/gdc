# Generative Distributional Control

<img src="https://i.imgur.com/U3KV0RI.png">

Generative Distributional Control (GDC) is a general framework for imposing constraints on samples of pretrained language models. The constraints can be either pointwise (e.g. all samples must be non-offensive) or distributional (e.g. a specified percentage of samples must mention females).

This repo contains code accompanying the following three papers:
* [`/dpg`](/dpg): [A Distributional Approach to Controlled Text Generation](https://arxiv.org/abs/2012.11635) (ICLR 2021)
* [`/cdpg`](/cdpg): [Controlling Conditional Language Models without Catastrophic Forgetting](https://arxiv.org/abs/2112.00791) (ICML 2022)
* [`/rm_vs_dm`](/rm_vs_dm): [On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting](https://arxiv.org/abs/2206.00761) (NeurIPS 2022)