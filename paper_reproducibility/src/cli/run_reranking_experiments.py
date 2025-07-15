import argparse
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import torch
import logging
from recbole.config import Config
from recbole.data import create_dataset
from tqdm import tqdm

from src.experiments_grid import (
    flatten_grid,
    generate_tags,
    load_grid_config,
    run_reranking_experiment_parallel,
)
from src.metrics import (
    entropy,
    intra_list_diversity,
    item_coverage,
    mean_inter_list_diversity,
    ndcg,
    precision,
    recall,
)
from src.rerankers import (
    dpp_rerank_user,
    mmr_rerank_user,
    no_rerank,
    sampled_mmr_rerank_user,
    ssd_rerank_user,
)
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    datasets_folder,
    dataset_name,
    scores_folder,
    model_name,
    use_text_embeddings,
    reranking_results_dir,
    candidates_top_k,
    rank_top_k,
    reranking_experiments_grid_path,
    reuse_previous_results,
):
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Scores folder: {scores_folder}")
    logging.info(f"Reranking results dir: {reranking_results_dir}")
    logging.info(f"Candidates top k: {candidates_top_k}")
    logging.info(f"Rank top k: {rank_top_k}")

    results_save_dir = Path(scores_folder) / dataset_name / model_name
    dataset_path = Path(datasets_folder) / dataset_name

    configs_dir = Path("configs") / dataset_name
    config = Config(
        config_file_list=[
            configs_dir / "base_config.yaml",
            configs_dir / f"{model_name}.yaml",
        ]
    )
    logging.info(f"Loading meta data of dataset")
    dataset = create_dataset(config)

    scores = torch.load(
        results_save_dir / "scores.pt", map_location=torch.device("cpu")
    )
    target_items = scores["target_items"]
    top_k_scores_values, top_k_scores_indicies = zip(*scores["predicted_items"])
    item_embeddings = scores["item_embeddings"]
    item_embeddings = np.array(item_embeddings)

    if use_text_embeddings:
        logging.info("Loading text embeddings")
        text_items_embeddings = torch.load(
            dataset_path / "text_items_embeddings.pt", map_location=torch.device("cpu")
        )
        item_embeddings = np.zeros_like(item_embeddings)
        for item_id, item_embedding in tqdm(
            zip(text_items_embeddings["item_id"], text_items_embeddings["embeddings"]),
            desc="Loading text embeddings",
        ):
            item_embeddings[item_id] = item_embedding


    # target_len = len(target_items)//10
    target_len = len(target_items)

    target_items = [x for x in target_items[:target_len] if len(x) > 0]
    target_items = [np.array(v) for v in target_items]
    top_k_scores_values = np.array(
        [x for x in top_k_scores_values[:target_len]  if len(x) > 0]
    )
    top_k_scores_indicies = np.array(
        [x for x in top_k_scores_indicies[:target_len]  if len(x) > 0]
    )

    logging.info(
        f"{len(target_items)=} {len(top_k_scores_values)=} {len(top_k_scores_indicies)=} {item_embeddings.shape=}"
    )

    candidates_top_k_logits, candidates_top_k_ids = (
        top_k_scores_values[:, :candidates_top_k],
        top_k_scores_indicies[:, :candidates_top_k],
    )

    # Reality check
    k = 200
    precision_value = precision(
        true_items=target_items, predicted_items=candidates_top_k_ids, k=k
    )
    recall_value = recall(
        true_items=target_items, predicted_items=candidates_top_k_ids, k=k
    )
    ndcg_value = ndcg(
        true_items=target_items, predicted_items=candidates_top_k_ids, k=k
    )

    item_coverage_value = item_coverage(
        predicted_items=candidates_top_k_ids[:100], k=k, total_items=dataset.item_num
    )
    mean_inter_list_diversity_value = mean_inter_list_diversity(
        predicted_items=candidates_top_k_ids[:100], k=k, total_items=dataset.item_num
    )
    intra_list_diversity_value = intra_list_diversity(
        predicted_items=candidates_top_k_ids[:100], k=k, item_embeddings=item_embeddings
    )
    entropy_value = entropy(
        predicted_items=candidates_top_k_ids[:100], k=k, total_items=dataset.item_num
    )

    logging.info("Reality check")
    logging.info(f"Precision@{k}: {precision_value:.4f}")
    logging.info(f"Recall@{k}: {recall_value:.4f}")
    logging.info(f"NDCG@{k}: {ndcg_value:.4f}")
    logging.info(f"Item Coverage@{k}: {item_coverage_value}")
    logging.info(f"Mean Inter-List Diversity@{k}: {mean_inter_list_diversity_value}")
    logging.info(f"Intra-List Diversity@{k}: {intra_list_diversity_value}")
    logging.info(f"Entropy@{k}: {entropy_value}")

    # Test reranking run to check if everything works correctly
    user = 1
    top_k = 200

    no_rerank_preds = no_rerank(
        pred_logits=candidates_top_k_logits[user],
        label_ids=candidates_top_k_ids[user],
        item_embeddings=item_embeddings,
        top_k=top_k,
    )

    mmr_reranked_preds = mmr_rerank_user(
        pred_logits=candidates_top_k_logits[user],
        label_ids=candidates_top_k_ids[user],
        item_embeddings=item_embeddings,
        top_k=top_k,
        lambda_=0.9,
    )

    dpp_reranked_preds = dpp_rerank_user(
        pred_logits=candidates_top_k_logits[user],
        label_ids=candidates_top_k_ids[user],
        item_embeddings=item_embeddings,
        top_k=top_k,
        alpha=0.7,
    )

    ssd_reranked_preds = ssd_rerank_user(
        pred_logits=candidates_top_k_logits[user],
        label_ids=candidates_top_k_ids[user],
        item_embeddings=item_embeddings,
        top_k=top_k,
        gamma=0.001,
    )

    smmr_reranked_preds = sampled_mmr_rerank_user(
        pred_logits=candidates_top_k_logits[user],
        label_ids=candidates_top_k_ids[user],
        item_embeddings=item_embeddings,
        top_k=top_k,
        lambda_=0.99,
        scale_factor=2,
        temperature=0.01,
    )

    logging.info("Items intersection:")
    logging.info(
        f"No rerank: {len(set(no_rerank_preds) & set(candidates_top_k_ids[user][:top_k]))} / {top_k}"
    )
    logging.info(
        f"MMR rerank: {len(set(mmr_reranked_preds) & set(candidates_top_k_ids[user][:top_k]))} / {top_k}"
    )
    logging.info(
        f"DPP rerank: {len(set(dpp_reranked_preds) & set(candidates_top_k_ids[user][:top_k]))} / {top_k}"
    )
    logging.info(
        f"SSD rerank: {len(set(ssd_reranked_preds) & set(candidates_top_k_ids[user][:top_k]))} / {top_k}"
    )
    logging.info(
        f"Sampled MMR rerank: {len(set(smmr_reranked_preds) & set(candidates_top_k_ids[user][:top_k]))} / {top_k}"
    )

    logging.info("Generating experiments grid")

    grid = load_grid_config(reranking_experiments_grid_path)
    flattened_experiments_grid = generate_tags(flatten_grid(grid))

    reranking_results_save_path = (
        Path(reranking_results_dir) / dataset_name / model_name
    )
    if (
        reuse_previous_results
        and (reranking_results_save_path / "reranking_results_.pt").exists()
    ):
        logging.info(f"Loading reranking results from {reranking_results_save_path}")
        previous_results = {
            item["tag"]: item
            for item in torch.load(
                reranking_results_save_path / "reranking_results_.pt"
            )["experiments_results"]
        }
        logging.info(f"Loaded {len(previous_results)} experiments")
        logging.info(f"Previous results: {list(previous_results.keys())}")
        for experiment in flattened_experiments_grid:
            if experiment["tag"] in previous_results:
                experiment.update(previous_results[experiment["tag"]])

    logging.info(
        f"Experiments grid: {[item['tag'] for item in flattened_experiments_grid]}"
    )
    time_since_last_save = time()
    for experiment in tqdm(flattened_experiments_grid):
        if experiment.get("result") is not None:
            logging.info(f"Skipping {experiment['tag']}")
            continue
        logging.info(f"Running {experiment['tag']}")
        experiment["result"] = run_reranking_experiment_parallel(
            experiment,
            candidates_top_k_logits,
            candidates_top_k_ids,
            item_embeddings,
            rank_top_k,
            verbose=False,
        )
        if time() - time_since_last_save > 60 * 30:
            logging.info("Saving reranking results")
            reranking_results_save_path.mkdir(exist_ok=True, parents=True)
            torch.save(
                {"experiments_results": flattened_experiments_grid},
                reranking_results_save_path / "reranking_results_.pt",
            )
            time_since_last_save = time()

    logging.info("Saving reranking results")
    reranking_results_save_path.mkdir(exist_ok=True, parents=True)
    torch.save(
        {"experiments_results": flattened_experiments_grid},
        reranking_results_save_path / "reranking_results_.pt",
    )

    metric_functions = [
        precision,
        recall,
        ndcg,
        item_coverage,
        mean_inter_list_diversity,
        intra_list_diversity,
        entropy,
    ]
    at_k_values = [10, 50, 100, 200]

    logging.info("Computing metrics")
    for experiment in tqdm(flattened_experiments_grid):
        if reuse_previous_results and experiment.get("metrics") is not None:
            logging.info(f"Skipping metrics computation for {experiment['tag']}")
            continue
        logging.info(f"Processing {experiment['tag']}")
        predicted_items = experiment["result"]
        metrics = []
        for metric_function in metric_functions:
            for at_k in at_k_values:
                metric_dict = {
                    "name": metric_function.__name__,
                    "at_k": at_k,
                    "value": metric_function(
                        true_items=target_items,
                        predicted_items=predicted_items,
                        k=at_k,
                        item_embeddings=item_embeddings,
                        total_items=dataset.item_num,
                    ),
                }
                metrics.append(metric_dict)
        experiment["metrics"] = metrics

    logging.info("Saving reranking + metrics results")
    reranking_results_save_path.mkdir(exist_ok=True, parents=True)
    torch.save(
        {"experiments_results": flattened_experiments_grid},
        reranking_results_save_path / "reranking_results_.pt",
    )

    for metric_function in metric_functions:
        fig = go.Figure()
        for experiment in flattened_experiments_grid:
            tag = experiment["tag"]
            metrics_df = pd.DataFrame(experiment["metrics"])
            metrics_df = metrics_df[metrics_df.name == metric_function.__name__]

            trace = go.Scatter(
                x=metrics_df["at_k"],
                y=metrics_df["value"],
                mode="lines+markers",
                name=f"{tag}",
                marker=dict(size=8),
                line=dict(dash="dash"),
                hovertemplate=(
                    f"<b>at_k:</b>" + " %{x}<br>"
                    f"<b>{metric_function.__name__}:</b>" + " %{y:.4f}<br>"
                    f"<b>tag:</b> {tag}<br>"
                ),
            )
            fig.add_trace(trace)

        fig.update_layout(
            title=f"{metric_function.__name__} in experiments; Dataset: {dataset_name}; Model: {model_name}",
            xaxis_title="at_k",
            yaxis_title=metric_function.__name__,
            xaxis=dict(tickmode="array", tickvals=sorted(metrics_df["at_k"].unique())),
            legend_title="Experiment tag",
            template="plotly",
            height=600,
            width=1200,
        )

        plots_path = reranking_results_save_path / "plotly_renders"
        plots_path.mkdir(exist_ok=True, parents=True)
        fig.write_html(plots_path / f"{metric_function.__name__}.html")

    metric_dfs = []
    for experiment in flattened_experiments_grid:
        tag = experiment["tag"]
        metrics_df = pd.DataFrame(experiment["metrics"])
        metrics_df["tag"] = tag
        metrics_df["algorithm"] = experiment["algorithm"].__name__
        metric_dfs.append(metrics_df)

    metrics_df = pd.concat(metric_dfs)
    metrics_df.to_csv(reranking_results_save_path / "metrics.csv")

    x_metrics = [
        "item_coverage",
        "mean_inter_list_diversity",
        "intra_list_diversity",
        "entropy",
    ]
    y_metrics = ["precision", "recall", "ndcg"]
    for x_metric in x_metrics:
        for y_metric in y_metrics:
            for at_k in at_k_values:
                fig = go.Figure()
                for algo in metrics_df.algorithm.unique():
                    subset = metrics_df[metrics_df.algorithm == algo]
                    subset = subset[subset.at_k == at_k]

                    trace = go.Scatter(
                        x=subset[subset.name == x_metric]["value"],
                        y=subset[subset.name == y_metric]["value"],
                        mode="markers",
                        name=f"{algo}",
                        marker=dict(size=8),
                        line=dict(dash="dash"),
                        hovertemplate=(
                            f"<b>{x_metric}:</b>" + " %{x}<br>"
                            f"<b>{y_metric}:</b>" + " %{y:.4f}<br>"
                            f"<b>algorithm:</b> {algo}<br>"
                        ),
                    )
                    fig.add_trace(trace)

                fig.update_layout(
                    title=f"Tradeoff between {x_metric} and {y_metric} in experiments; @k: {at_k}; Dataset: {dataset_name}; Model: {model_name}",
                    xaxis_title=x_metric,
                    yaxis_title=y_metric,
                    xaxis=dict(tickmode="array"),
                    legend_title="Algorithm",
                    template="plotly",
                    height=600,
                    width=1000,
                )

                plots_path = (
                    reranking_results_save_path / "tradeoff_plots" / "plotly_renders"
                )
                plots_path.mkdir(exist_ok=True, parents=True)
                fig.write_html(plots_path / f"{x_metric}-{y_metric}@{at_k}.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reranking experiments for recommendation systems."
    )

    parser.add_argument(
        "--datasets_folder",
        type=str,
        default="dataset",
        help="Folder containing datasets.",
    )

    parser.add_argument(
        "--dataset_name", type=str, default="ml-100l", help="Name of the dataset."
    )
    parser.add_argument(
        "--scores_folder",
        type=str,
        default="training_results",
        help="Folder containing the scores.",
    )
    parser.add_argument(
        "--model_name", type=str, default="BPR", help="Name of the model."
    )
    parser.add_argument(
        "--use_text_embeddings",
        action="store_true",
        default=False,
        help="Whether to use text embeddings of items",
    )

    parser.add_argument(
        "--reranking_results_dir",
        type=str,
        default="reranking_results",
        help="Directory to save reranking results.",
    )
    parser.add_argument(
        "--candidates_top_k",
        type=int,
        default=1000,
        help="Number of top candidates to consider.",
    )
    parser.add_argument(
        "--rank_top_k",
        type=int,
        default=200,
        help="Number of top items to rank.",
    )
    parser.add_argument(
        "--reranking_experiments_grid_path",
        type=str,
        default="reranking_experiments_grid.yaml",
        help="Path to the reranking experiments grid.",
    )
    parser.add_argument(
        "--reuse_previous_results",
        action="store_true",
        default=False,
        help="Whether to reuse previous results.",
    )

    args = parser.parse_args()
    logging.info(f"Args: {args}")

    main(
        datasets_folder=args.datasets_folder,
        dataset_name=args.dataset_name,
        scores_folder=args.scores_folder,
        model_name=args.model_name,
        use_text_embeddings=args.use_text_embeddings,
        reranking_results_dir=args.reranking_results_dir,
        candidates_top_k=args.candidates_top_k,
        rank_top_k=args.rank_top_k,
        reranking_experiments_grid_path=args.reranking_experiments_grid_path,
        reuse_previous_results=args.reuse_previous_results,
    )
