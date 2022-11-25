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

## Contributors

<img width=35px src="https://tomekkorbak.com/images/fotka.jpg"> [Tomasz Korbak](https://tomekkorbak.com), tomasz.korbak@gmail.com

<img width=35px src="https://i.imgur.com/YLwtSt1.png"> [Hady Elsahar](http://hadyelsahar.io), hady.elsahar@naverlabs.com

<img width=35px src="https://pbs.twimg.com/profile_images/1268967379758915593/q0ofGxv9_400x400.jpg"> [Germán Kruszewski](https://germank.github.io/), german.kruszewski@naverlabs.com

<img width=35px src="https://i.imgur.com/gqHaXeG.png"> [Marc Dymetman](https://scholar.google.com/citations?user=D6J5pooAAAAJ&hl=en), marc.dymetman@naverlabs.com


## Citation

```bibtex
@InProceedings{pmlr-v162-korbak22a,
  title = 	 {Controlling Conditional Language Models without Catastrophic Forgetting},
  author =       {Korbak, Tomasz and Elsahar, Hady and Kruszewski, German and Dymetman, Marc},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {11499--11528},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/korbak22a/korbak22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/korbak22a.html},
}
```
