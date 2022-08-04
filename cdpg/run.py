import os
import neptune
import json
import torch
import wandb
import time
from itertools import chain

from tqdm import tqdm as tqdm
import numpy as np
from pprint import pprint
tqdm.pandas()

from cdpg.core import postprocess_code_generation
from cdpg.transformers_wrappers import T5ForConditionalGenerationWrapper, GPT2Wrapper, GPTNeoWrapper
from cdpg.pointwise_gdc import PointwiseGDCTrainer
from cdpg.metrics import Distinct_N, SelfBlEU, SacreBLEU, Rouge, BertScore, BLEURT, CharLength, TokenLength,\
    WordLength, NamedEntities, PEP8Errors, ASTNodeCount, Repetitions, UpvoteScore, Specificity, Compilability
from cdpg.pg import PGTrainer
from cdpg.ppo import PPOTrainer
import logging
import argparse
from transformers import GPT2Tokenizer, T5TokenizerFast
import pandas as pd

from cdpg.scorer import Scorer


def pretty_print(log_dict):
    for k,v in log_dict.items():
        print("\t\t{0: <20}:{1: <20}".format(k,v))


def to_jsonl(log_dict):
    d = dict()
    for k,v in log_dict.items():
        v = float(v)
        d[k] = v
    d = json.dumps(d)
    return d


def log_text_to_wandb_neptune_file(game_data, config, basename, wandb_run, neptune_experiment):
    game_data_df = pd.DataFrame(game_data)
    game_data_df.rename(columns={"scores":"b(x)"}, inplace=True)

    log_cols = ["epoch", "b(x)", "query", "response"]
    if "reference" in game_data_df:
        log_cols += ["reference"]
    game_data_df = game_data_df[log_cols]

    ## log to file system
    file_path = config['save_dir'] + f'/{basename}'

    # append to a csv file
    game_data_df.to_csv(file_path+".csv", mode="a", header=False, index=False)
    # also append to a text file in a readable format
    with open(file_path+".txt", 'a+', encoding='utf-8') as f:
        for r in game_data_df.to_dict("records"):
            f.writelines([f'{k}:\t{v}\n' for k,v in r.items()])
        f.write("\n-----------\n")

    # log to neptune and wandb
    # currently neptune doesn't allow lots of logs to keep max of 20
    n = config.get("NEPTUNE_LOG_SAMPLE_SIZE", 16)
    n_game_data = game_data_df.sample(n)  # randomly sample max values

    if neptune_experiment:
        n_game_data.to_csv("samples_tmp.csv",index=False)
        epoch = n_game_data["epoch"].values[0]
        neptune_experiment.log_artifact('samples_tmp.csv', f'{basename}_{epoch}.csv')

        for r in n_game_data.to_dict('records'):
            s = "|".join(["{}: {}".format(k,v) for k,v in r.items()])[:900]
            neptune_experiment.log_text(log_name=f'{basename}', x=s, timestamp=time.time())

    if wandb_run:
        wandb_run.log({f"{basename}": wandb.Table(dataframe=n_game_data)})

    return


def load_neptune(config):
    if os.environ.get("NEPTUNE_PROJECT_NAME", ""):
        assert "NEPTUNE_API_TOKEN" in os.environ, "If using Neptune, you must set NEPTUNE_API_TOKEN"
        print("Neptune: Initializing...")
        experiment_id = config.get("run_id") or os.environ.get('SLURM_JOB_ID')
        config['path'] = os.getcwd()
        proxies = {
            'http': '',
            'https': '',
        }
        project = neptune.init(os.environ["NEPTUNE_PROJECT_NAME"], proxies=proxies)
        if experiment_id and len(project.get_experiments(tag=experiment_id)) > 0:
            experiment = project.get_experiments(tag=experiment_id)[0]
            print(f"Neptune: Found existing experiment: {experiment_id}")
        else:
            tags = config.get("WANDB_NEPTUNE_TAGS", [])
            experiment = project.create_experiment(
                name=experiment_id,
                description=config.get("WANDB_NEPTUNE_DESC", ""),
                tags=tags,
                upload_source_files=[],
                params=config)
            print("Neptune: Experiment created")
        return experiment


def load_wandb(config):
    if os.environ.get("WANDB_PROJECT_NAME", ""):
        assert "WANDB_API_KEY" in os.environ, "If using Wandb, you must set WANDB_API_KEY"
        print("logging to wandb..")
        run_object = wandb.init(project=os.environ["WANDB_PROJECT_NAME"], config=config,
                    tags=config.get("WANDB_NEPTUNE_TAGS", []),
                    notes=config.get("WANDB_NEPTUNE_DESC", ""),
                    resume="allow",
                    id=config.get("run_id") or os.environ.get('SLURM_JOB_ID')
        )
        return run_object


def sample_and_score(model, tokenizer, scoring_function, empty_prefix, top_p=1.0, sample_size=None,
                     force_condition=None, params=None, num_samples_per_query=1, decoding_params=None):
    """
    Samples a minibatch from model and scores it with appropriate b(x) model
    """
    device = next(model.parameters()).device
    timing = dict()

    # if sample size is not given sample 1 batch
    if sample_size is None:
        sample_size = params["batch_size"]
    assert sample_size % num_samples_per_query == 0
    num_queries = int(sample_size/num_samples_per_query)
    file_name = force_condition or params['query_set_train']
    prefix = model.get_prefix(task_name=params['task'])
    print(f'Sampling from queryset {file_name} with prefix "{prefix}"')
    if not hasattr(model, 'cached_queries'):
        model.cached_queries = {}

    if file_name not in model.cached_queries:
        df = pd.read_csv(file_name).dropna()
        assert "source" in df, f'no source column in {file_name}'
        df["query"] = df["source"]  ## legacy since code uses query not source
        model.cached_queries[file_name] = df

    game_data_df =  model.cached_queries[file_name].sample(n=num_queries, replace=True)

    if params.get('Z_c') == 'oracle':
        assert "Z_c" in model.cached_queries[file_name], \
            "Z_c='oracle' param in config necessitates Z_c values should be given as a column in {file_name}"
        # adding epsilon on Z_c values
        game_data_df["Z_c"] += 1e-4

    game_data = game_data_df.to_dict(orient="list")
    # repeat each query, reference, and other meta data by num_samples_per_query 
    for k, v in game_data.items():
        game_data[k] = list(chain(*([v1] * num_samples_per_query for v1 in v)))

    assert len(game_data['query']) == sample_size

    # adding task prefix to queries e.g. "Translate en to fr"  for T5 models
    queries_prefixed = [prefix + q for q in game_data["query"]]

    tokenizer_queries = tokenizer(queries_prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt')
    query_tensors = tokenizer_queries['input_ids'].to(device)
    query_masks = tokenizer_queries['attention_mask'].to(device)

    t = time.time()
    response_tensors = []
    response_masks = []

    fbs = params['forward_batch_size']
    for i in range(int(sample_size/fbs)):
        # sampling response for each input query
        # generate responses using q(x), which is gpt2_model_ref
        response, response_mask = model.respond_to_batch(
            input_ids=query_tensors[i*fbs:(i+1)*fbs],
            attention_mask=query_masks[i*fbs:(i+1)*fbs],
            top_p=top_p,
            decoding_params=decoding_params or {'do_sample': True}
        )
        response_tensors.append(response)
        response_masks.append(response_mask)

    response_tensors = torch.cat(response_tensors)
    response_masks = torch.cat(response_masks)

    # decoding them to get text
    game_data['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=config['task'] != 'code_generation_old')

    if config['task'] in ['code_generation']:
        game_data['response'] = [postprocess_code_generation(response) for response in game_data['response']]

    timing['time/generation_fn'] = time.time() - t

    # calculate scores
    t = time.time()
    scores = []

    dfbs = params.get('discriminator_forward_batch_size', fbs) # use disc. fbs if available
    for i in range(int(sample_size / dfbs)):
        # check for presence of the word beautiful
        bs_game_data = {}
        for k, v in game_data.items():
            bs_game_data[k] = game_data[k][i*dfbs:(i+1)*dfbs]

        additional_data = {k: v for k, v in bs_game_data.items() if k not in ["response", "query", "source"]}
        res = scoring_function(continuations=bs_game_data["response"], queries=bs_game_data["query"], **additional_data)
        scores.append(res)

    # sometimes b(x) is not supposed to be calculated
    # This case solely happens in the evaluation case only
    # e.g. evaluation of GenderBiased Dataset on a separate dataset, scores are returned = None
    if scores[0] is None:
        scores = None
    else:
        scores = torch.cat(scores)
    timing['time/scoring_fn'] = time.time()-t
    return game_data, timing, query_tensors, response_tensors, query_masks, response_masks, scores


def main(config):
    neptune_experiment = load_neptune(config)
    wandb_run = load_wandb(config)
    tokenizer_class = eval(config['tokenizer_class'])
    tokenizer = tokenizer_class.from_pretrained(config['tk_name'], **config.get('tokenizer_kwargs', {}))
    if config['tokenizer_class'] == 'GPT2Tokenizer':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    scorer = Scorer(**config)
    scoring_fn = scorer.get_scoring_fn()

    logging.info("Creating {} Trainer...".format(config['trainer_class']))
    trainer_cls = eval(config['trainer_class'])

    # initialized trainer
    trainer = trainer_cls(
        model_cls=eval(config['model_class']),
        tokenizer=tokenizer,
        sampling_function=sample_and_score,
        scoring_function=scoring_fn,
        **config
    )
    if config['task'] in ['dialogue', 'code_generation']:
        if config['model_class'] == 'GPT2HeadWithValueModel':
            trainer.orig_model.config.pad_token_id = trainer.orig_model.config.eos_token_id
            trainer.model.config.pad_token_id = trainer.model.config.eos_token_id
        if config['model_class'] in ['GPT2Wrapper', 'GPTNeoWrapper']:
            trainer.orig_model.model.config.pad_token_id = trainer.orig_model.model.config.eos_token_id
            trainer.model.model.config.pad_token_id = trainer.model.model.config.eos_token_id

    # check if resuming is allowed and checkpoint exists
    last_ckpt_dir = os.path.join(config['save_dir'], 'checkpoint_last.pt')

    start_epoch = 0
    if config['resume_if_ckpt_exists'] and os.path.exists(last_ckpt_dir):
        print("Resuming training from {}".format(last_ckpt_dir))
        trainer.load_checkpoint(last_ckpt_dir)
        start_epoch = trainer.iter # load last epoch

    ###############################
    ##### Evaluation Metrics ######
    ###############################
    # initialize metrics
    dists = [Distinct_N(n) for n in [1, 2, 3]]
    self_bleus = [SelfBlEU(gram=n) for n in [3, 4, 5]]
    metrics = [*dists, *self_bleus, CharLength()]
    if config['task'] == 'summarization':
        metrics += [
            Rouge(),
            BertScore(batch_size=config['forward_batch_size'], device=config.get('gpt2_orig_device', 'cpu')),
            NamedEntities(),
            WordLength(),
            TokenLength(tokenizer)
        ]
    if 'translation' in config['task']:
        metrics += [SacreBLEU()]
    if config['task'] in ['code_generation']:
        metrics += [
            Compilability(),
            PEP8Errors(),
            ASTNodeCount()
        ]
    if config['task'] == 'dialogue':
        metrics += [
            SacreBLEU(),
            Repetitions(n=1),
            Repetitions(n=2),
            UpvoteScore(),
            Specificity(word2idf_path='../cdpg/resources/personachat_word2idf.json')
        ]

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

        game_data, timing, query_tensors, response_tensors, query_masks, response_masks, scores = sample_and_score(
            trainer.get_sampling_model(),
            tokenizer,
            scoring_fn,
            config['empty_prefix'],
            params=config,
            num_samples_per_query=config.get('num_samples_per_query_for_train', 1),
            decoding_params=config.get('train_decoding_params')
        )

        assert scores is not None, "b(x) is not evaluated make sure all score functions are provided in the input csv query file"
        game_data["scores"] = [i.item() for i in scores]
        game_data["epoch"] = [epoch]* len(scores)

        if config.get('Z_c') == 'MC':
            trainer.compute_Z_c_using_MC(
                query_tensors,
                response_tensors,
                query_masks,
                response_masks,
                game_data,
                scores,
                num_samples_per_query=config.get('num_samples_per_query_for_train', 1)
            )

        # =================== #
        #       Training      #
        # =================== #
        stats, step_logs = trainer.step(query_tensors, response_tensors, query_masks, response_masks, scores, game_data)

        # =================== #
        #       Logging       #
        # =================== #
        train_logs.update(step_logs)

        # logging generated examples to wandb neptune and filesystem
        basename = "generations_train_sampling"
        log_text_to_wandb_neptune_file(game_data, config, basename, wandb_run, neptune_experiment)

        train_logs["epoch"] = epoch
        train_log_file.write(to_jsonl(train_logs) + '\n')
        train_log_file.flush()

        all_logs = train_logs
        # policy evaluation
        if config.get('eval', True) and epoch % config.get('eval_interval') == 0:
            eval_logs = dict()
            for query_set_name in config['query_sets_eval']:
                for eval_mode in ['sampling', 'map']:
                    query_set_name_short = query_set_name.split('/')[-1]
                    if config['task'] == 'lm_debiasing' and 'null' in query_set_name:
                        cond_eval_scoring_fn = Scorer(
                            scorer_type='sentiment',
                            gpt2_sentiment_device=config['gpt2_sentiment_device']
                        ).get_scoring_fn()
                    else:
                        cond_eval_scoring_fn = scoring_fn
                    # if eval_sample_size is no given set it to config bsz
                    config["eval_sample_size"] = config.get("eval_sample_size", config["batch_size"])

                    game_data, _, _, _, query_masks, response_masks, scores = sample_and_score(
                        trainer.get_eval_model(),
                        tokenizer,
                        cond_eval_scoring_fn,
                        config['empty_prefix'],
                        top_p=config.get('eval_top_p', 1.0),
                        sample_size=config["eval_sample_size"],
                        params=config,
                        num_samples_per_query=config.get('num_samples_per_query_for_eval'),
                        decoding_params=config.get('eval_decoding_params') if eval_mode == 'map' else config.get('train_decoding_params'),
                        force_condition=query_set_name
                    )

                    game_data["epoch"] = [epoch]* len(game_data["response"])

                    # sometimes b(x) is not meant to be calculated during evaluation
                    # in this case the sample_and_score returns None value for scores
                    if scores is None:
                        game_data["scores"] = ["None"] * len(game_data["response"])
                    else:
                        game_data["scores"] = [i.item() for i in scores]
                        eval_logs[f'Eval/{query_set_name_short}/{eval_mode}/b(x)_mean'] = scores.mean().item()

                    # logging generated examples to wandb neptune and filesystem
                    basename = "generations_{}_{}_{}".format("eval",query_set_name_short[:-4], eval_mode)
                    log_text_to_wandb_neptune_file(game_data, config, basename, wandb_run, neptune_experiment)

                    # log metrics
                    for m in metrics:
                        # if the metric requires a reference and a reference is not provided 
                        if not m.referenceless and "reference" not in game_data:
                            print("WARNING: {} metric cannot be calculated on {} as it doesn't have reference sentences".format(m.name, query_set_name))
                        else:
                            # game_data contains "response", "query",
                            metric_scores = m.compute_metric(**game_data)
                            if isinstance(metric_scores, dict):
                                eval_logs.update({f'Eval/{query_set_name_short}/{eval_mode}/{k}': v for k, v in metric_scores.items()})
                            else:
                                eval_logs[f"Eval/{query_set_name_short}/{eval_mode}/{m.get_name()}"] = metric_scores

                if game_data["scores"][0] != "None":  # check if b(x) is able to be computed to calculate KL metrics otherwise skip
                    eval_logs[f"Eval/{query_set_name_short}/KL(p || pi)"] = trainer.eval_kl_p(force_condition=config.get('query_set_KL', config['query_set_train']))
                    eval_logs[f"Eval/{query_set_name_short}/KL(pi || a)"] = trainer.eval_kl_a(force_condition=config.get('query_set_KL', config['query_set_train']))
            eval_logs["epoch"] = epoch
            eval_log_file.write(to_jsonl(eval_logs) + '\n')
            eval_log_file.flush()  # write eval logs to file
            all_logs.update(eval_logs)
        all_logs.update(timing)
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
    parser.add_argument('-o', '--override', nargs='+', type=lambda x: (x.split("=")[0], "=".join(x.split("=")[1:])),
                        default={}, help="usage: -o param1=v1 param2=v2")

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

    override_dict = {k: parse(k, v) for k, v in args.override}
    config.update(override_dict)  # override args

    if 'gpt2_device' not in config:
        config['gpt2_device'] = 0
    if 'gpt2_ref_device' not in config:
        config['gpt2_ref_device'] = 1
    if 'gpt2_orig_device' not in config:
        config['gpt2_orig_device'] = 1

    # create checkpoint folder if not exists
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    pprint(config)
    config['path'] = os.getcwd()
    # log the config into the same directory as the training log
    with open(os.path.join(config.get('save_dir'), "config.json"), "w") as f:
        f.write(json.dumps(config))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    main(config)
