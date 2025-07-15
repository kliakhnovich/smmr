from itertools import product

from src.rerankers import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import yaml


def load_grid_config(grid_config_path):
    with open(grid_config_path, "r") as file:
        experiments_grid = yaml.safe_load(file)
    for item in experiments_grid:
        item["algorithm"] = eval(item["algorithm"])
    return experiments_grid


def flatten_grid(grid):
    experiment_args = []
    for config in grid:
        params = config["params"]
        params_combinations = [
            {"params": dict(zip(params.keys(), params_values))}
            for params_values in list(product(*params.values()))
        ]
        if len(params_combinations) == 0:
            params_combinations = [{}]
        for v in params_combinations:
            v.update({"algorithm": config["algorithm"]})
        experiment_args.extend(params_combinations)
    return experiment_args


def generate_tags(flat_grid):
    for config in flat_grid:
        tag = "_".join(
            [
                config["algorithm"].__name__,
                *[f"{k}_{v}" for k, v in config["params"].items()],
            ]
        )
        config["tag"] = tag
    return flat_grid


def process_user(args):
    """
    Function to process an individual user's reranking.
    """
    (
        user_idx,
        candidates_logits,
        candidates_ids,
        item_embeddings,
        experiment_args,
        top_k,
    ) = args
    reranked_items = experiment_args["algorithm"](
        pred_logits=candidates_logits,
        label_ids=candidates_ids,
        item_embeddings=item_embeddings,
        top_k=top_k,
        **experiment_args["params"],
    )
    return reranked_items


def run_reranking_experiment_parallel(
    experiment_args,
    candidates_top_k_logits,
    candidates_top_k_ids,
    item_embeddings,
    top_k=200,
    verbose=False,
):
    all_reranked_users = []

    # Create arguments for each user
    job_args = [
        (
            user_idx,
            candidates_top_k_logits[user_idx],
            candidates_top_k_ids[user_idx],
            item_embeddings,
            experiment_args,
            top_k,
        )
        for user_idx in range(len(candidates_top_k_logits))
    ]

    # Initialize progress bar if verbose
    if verbose:
        job_args = tqdm(job_args, total=len(candidates_top_k_logits))

    # Initialize multiprocessing pool
    with Pool(cpu_count() - 1) as pool:
        # Map the jobs and store results in parallel
        all_reranked_users = pool.map(process_user, job_args)

    return all_reranked_users
