# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0

import os
import neptune
import json
import torch
import wandb
import time
import sys
import string

from tqdm import tqdm as tqdm
import numpy as np
from pprint import pprint
tqdm.pandas()

sys.path.append("../")
from gdc.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from gdc.pointwise_gdc import PointwiseGDCTrainer
from gdc.metrics import Distinct_N, SelfBlEU
from gdc.pg import PGTrainer
from gdc.ppo import PPOTrainer
import logging
import argparse
from transformers import GPT2Tokenizer

from gdc.scorer import Scorer


printable = set(string.printable)


def clean_str(s):
    return ''.join(filter(lambda x: x in printable, s))


def pretty_print(log_dict):
    for k,v in log_dict.items():
        print("\t\t{0: <20}:{1: <20}".format(k,v))


def to_ljson(log_dict):
    d = dict()
    for k,v in log_dict.items():
        v = float(v)
        d[k] = v
    d = json.dumps(d)
    return d


def load_neptune(config):
    if config["NEPTUNE"]:
        assert "WANDB_NEPTUNE_NAME" in config, "You must specify a name for the experiment"
        os.environ['NEPTUNE_API_TOKEN'] = config["NEPTUNE_API_TOKEN"]
        print("Neptune: Initializing..")
        experiment_id = config.get("run_id", None)
        project = neptune.init(config["NEPTUNE_PROJECT_NAME"])
        if experiment_id and len(project.get_experiments(tag=experiment_id)) > 0:
            experiment = project.get_experiments(tag=experiment_id)[0]
            print(f"Neptune: Found existing experiment: {experiment_id}")
        else:
            experiment = project.create_experiment(
                name=config["WANDB_NEPTUNE_NAME"],
                description=config.get("WANDB_NEPTUNE_DESC", ""),
                tags=config.get("WANDB_NEPTUNE_TAGS", []) + [experiment_id],
                upload_source_files=[],
                params=config)
            print("Neptune: Experiment created")
        return experiment


def load_wandb(config):
    if config["WANDB"]:
        print("logging to wandb..")
        run_object = wandb.init(project=config["WANDB_PROJECT_NAME"], config=config,
                    tags=config.get("WANDB_NEPTUNE_TAGS", []),
                    notes=config.get("WANDB_NEPTUNE_DESC", ""),
                    resume="allow", id=config.get("run_id", None)
                )
        return run_object


def sample_and_score(model, tokenizer, scoring_function, empty_prefix, top_p=1.0, sample_size=None):
    """
    Samples a minibatch from model and scores it with appropriate b(x) model
    """
    device = next(model.parameters()).device
    game_data = dict()
    timing = dict()

    # if sample size is not given sample 1 batch
    if sample_size is None:
        sample_size = config["batch_size"]

    # Prepare queries: fill queries with BOS token
    if empty_prefix:
        prefix_str= tokenizer.eos_token
    else:
        prefix_str= tokenizer.eos_token + config["prefix"]

    game_data['query'] = [prefix_str] * sample_size
    query_tensors = torch.stack([torch.LongTensor(tokenizer.encode(prefix_str))] * sample_size).to(device)

    t = time.time()
    response_tensors = []

    fbs = config['forward_batch_size']
    for i in range(int(sample_size/fbs)):
        # sampling response for each input query
        # generate responses using q(x), which is gpt2_model_ref 

        response = respond_to_batch(model, query_tensors[i*fbs:(i+1)*fbs].to(device),
                                     txt_len=config['txt_out_len'], top_p=top_p)
        response_tensors.append(response)

    response_tensors = torch.cat(response_tensors)

    # decoding them to get text
    game_data['response'] = [tokenizer.decode(response_tensors[i, :]) for i in range(sample_size)]
    timing['time/get_response'] = time.time() - t

    # calculate scores
    t = time.time()
    scores = []

    dfbs = config.get('discriminator_forward_batch_size', fbs) # use disc. fbs if available
    for i in range(int(sample_size / dfbs)):
        # check for presence of the word beautiful
        responses = game_data['response'][i*dfbs:(i+1)*dfbs]
        res= scoring_function(responses)
        scores.append(res)

    scores = torch.cat(scores)
    timing['time/get_sentiment_preds'] = time.time()-t

    return game_data, timing, query_tensors, response_tensors, scores


def main(config):
    neptune_experiment = load_neptune(config)
    wandb_run = load_wandb(config)

    #tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])
    scorer = Scorer(**config)
    scoring_fn = scorer.get_scoring_fn()

    logging.info("Creating {} Trainer...".format(config['trainer_class']))
    trainer_cls = eval(config['trainer_class'])

    # initialized trainer
    trainer = trainer_cls(model_cls=GPT2HeadWithValueModel,
                            tokenizer=gpt2_tokenizer,
                            sampling_function=sample_and_score, 
                            scoring_function=scoring_fn, 
                            **config)

    #### check if resuming is allowed and checkpoint exists
    last_ckpt_dir = os.path.join(config['save_dir'], 'checkpoint_last.pt')

    start_epoch = 0
    if config['resume_if_ckpt_exists'] and os.path.exists(last_ckpt_dir):
        print("Resuming training from {}".format(last_ckpt_dir))
        trainer.load_checkpoint(last_ckpt_dir)
        start_epoch = trainer.iter # load last epoch

    # initialize metrics
    dists = [Distinct_N(n) for n in [1, 2, 3]]
    self_bleus = [SelfBlEU(gram=n) for n in [3, 4, 5]]
    metrics = [*dists, *self_bleus]

    # create checkpoint folder if not exists
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    tr_log_file = config["save_dir"] + "/train_log.json"
    ev_log_file = config["save_dir"] + "/eval_log.json"
    if os.path.exists(tr_log_file):
        print("Appending to log file...")
    train_log_file = open(tr_log_file, "a")  # append to file if exists
    eval_log_file = open(ev_log_file, "a")

    for epoch in tqdm(range(start_epoch, int(np.ceil(config["steps"]/config['batch_size'])), 1)):
        print("Epoch {}:".format(epoch))
        torch.cuda.empty_cache()
        train_logs = dict()

        game_data, timing, query_tensors, response_tensors,  scores = sample_and_score(
            trainer.get_sampling_model(),
            gpt2_tokenizer,
            scoring_fn,
            config['empty_prefix']
        )

        # =================== #
        #       Training      #
        # =================== #
        stats, step_logs = trainer.step(query_tensors, response_tensors, scores)

        # =================== #
        #       Logging       #
        # =================== #

        train_logs.update(step_logs)

        # log some samples
        samples = [[i, k, j] for i, k, j in zip([epoch]*len(scores), scores, game_data["response"])]
        log_samples = samples[:config.get("NEPTUNE_LOG_SAMPLE_SIZE", 20)]

        if neptune_experiment:
            for i, k, j in log_samples:
                r = "epoch:{} \t b(x)={} \t : {}".format(i, k, j)
                neptune_experiment.log_text(log_name='Train/samples', x=clean_str(r), timestamp=time.time())

        if wandb_run:
            wandb_run.log({"Train/samples": wandb.Table(data=log_samples, columns=["epoch", "b(x)", "Text"])})

        train_logs["epoch"] = epoch
        train_log_file.write(to_ljson(train_logs) + '\n')
        train_log_file.flush()

        all_logs = train_logs
        # policy evaluation
        if config.get('eval', True) and (epoch+1) % config.get('eval_interval') == 0:
            # if eval_sample_size is no given set it to config bsz
            config["eval_sample_size"] = config.get("eval_sample_size", config["batch_size"])

            print("Evaluating with nucleus sampling...")
            game_data, _, _, _, scores = sample_and_score(
                trainer.get_eval_model(),
                gpt2_tokenizer, scoring_fn,
                config['empty_prefix'],
                top_p=config.get('eval_top_p', 0.9),
                sample_size=config["eval_sample_size"]
            )

            eval_logs= dict()
            eval_logs['Eval/b(x)_mean'] = scores.mean().item()

            # log some samples
            samples = [[i, k, j] for i, k, j in zip([epoch]*len(scores), scores, game_data["response"])]

            log_samples = samples[:config.get("NEPTUNE_LOG_SAMPLE_SIZE", 20)]
            if neptune_experiment:
                for i, k, j in log_samples:
                    r = "epoch:{} \t b(x)={} \t : {}".format(i, k, j)
                    neptune_experiment.log_text(log_name='Eval/samples', x=clean_str(r), timestamp=time.time())

            if wandb_run:
                wandb_run.log({"Eval/sampelf.params['q_update_interval'] * les": wandb.Table(data=log_samples,
                                        columns=["epoch", "b(x)", "Text"])})

            # sample from Ref model
            eval_logs["Eval/KL(p || pi)"] = trainer.eval_kl_p()
            # sample from eval model
            eval_logs["Eval/KL(pi || a)"] = trainer.eval_kl_a()

            # log metrics
            for m in metrics:
                eval_logs["Eval/" + m.get_name()] = m.compute_metric(texts=game_data["response"])
            
            eval_logs["epoch"] = epoch
            eval_log_file.write(to_ljson(eval_logs) + '\n')
            eval_log_file.flush()  # write eval logs to file
            all_logs.update(eval_logs)

        pretty_print(all_logs)  # pretty print step logs

        if neptune_experiment:
            for k, v in all_logs.items():
                neptune_experiment.log_metric(k, v)

        if wandb_run:
            wandb_run.log(all_logs)

        if (epoch+1) % config['save_checkpoint_every'] == 0:
            print("saving checkpoint to {}".format(config["save_dir"]))
            trainer.save_checkpoint(config['save_dir'])

        
    trainer.save_checkpoint(config['save_dir'])
    train_log_file.close()
    eval_log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run code for EBM Control')
    parser.add_argument("--config", dest="config", required=True,
                        help="config file for experiment params", metavar="FILE")
    parser.add_argument('-o', '--override', nargs='+', type=lambda x: (x.split("=")[0],"=".join(x.split("=")[1:])),
                        default={},help="usage: -o param1=v1 param2=v2")

    args = parser.parse_args()

    # load config file
    config = json.loads(open(args.config).read())

    # override args
    def parse(k, x):
        if k in ["trainer_class"]:  # exception params to treat as text
            return x
        try:
            return eval(x)  # int, array, float, quoted string
        except:
            return x

    override_dict = {k: parse(k, v) for k,v in args.override}
    config.update(override_dict)  # override args

    device_1, device_2 = 0, 1

    config['gpt2_device'] = device_1
    config['gpt2_orig_device'] = config['gpt2_sentiment_device'] = config['gpt2_ref_device'] = device_2

    config["save_log_dir"] = "../saved_logs"

    pprint(config)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    main(config)
