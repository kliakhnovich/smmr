import torch.distributed as dist
import torch

if torch.cuda.is_available():
    dist.init_process_group(backend="nccl")
else:
    dist.init_process_group(backend="gloo")

import argparse

from pathlib import Path

import numpy as np
import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, LightGCN
from recbole.model.sequential_recommender import SASRec
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils.case_study import full_sort_scores

from src.utils import get_new_sequence_mask_np
from tqdm import tqdm

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    dataset_name,
    model_name,
    top_k_scores_cutoff,
    configs_dir,
    results_save_dir,
    force_train,
):
    logging.info(f"Running {model_name} on {dataset_name}")
    results_save_dir = Path(results_save_dir) / dataset_name / model_name
    results_save_dir.mkdir(exist_ok=True, parents=True)

    config = Config(
        config_file_list=[
            Path(configs_dir) / dataset_name / "base_config.yaml",
            Path(configs_dir) / dataset_name / f"{model_name}.yaml",
        ]
    )
    print(config)

    if not force_train and (results_save_dir / "model.pt").exists():
        logging.info("Loading pre-trained model")
        _, model, dataset, train_data, valid_data, _ = load_data_and_model(
            model_file=results_save_dir / "model.pt"
        )
        print(config)
        print(dataset)
    else:
        logging.info("Training model from scratch")
        dataset = create_dataset(config)
        print(dataset)
        train_data, valid_data, _ = data_preparation(config, dataset)
        model = eval(config.model)(config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)
        trainer.saved_model_file = results_save_dir / "model.pt"
        trainer.fit(train_data, valid_data=valid_data, show_progress=True, verbose=True)
        valid_results = trainer.evaluate(valid_data)
        logging.info(f"{valid_results=}")

    if isinstance(model, SASRec):
        logging.info(f"calculating scores for {model.__class__.__name__}")
        first_sequence_mask = get_new_sequence_mask_np(
            valid_data.dataset.inter_feat.interaction[config["USER_ID_FIELD"]]
        )
        interactions = valid_data.dataset.inter_feat[first_sequence_mask]
        user_ids = interactions[config["USER_ID_FIELD"]].numpy()
        device = "cpu"
        model = model.to(device).eval()
        with torch.no_grad():
            scores = model.full_sort_predict(interactions.to(device))
            scores[:, 0] = -torch.inf
            scores = scores.cpu()
        logging.info(f"scores calculated, shape: {scores.shape}")

        top_k_scores_values, top_k_scores_indicies = scores.topk(
            min(top_k_scores_cutoff, scores.shape[1])
        )

        # fill missing users
        preds_dict = dict(
            zip(user_ids, zip(top_k_scores_values, top_k_scores_indicies))
        )
        true_items_for_user = []
        predicted_items = []
        for user in tqdm(range(dataset.user_num)):
            true_items_for_user.append(
                valid_data.dataset.inter_feat[config["ITEM_ID_FIELD"]][
                    valid_data.dataset.inter_feat[config["USER_ID_FIELD"]] == user
                ]
            )
            predicted_items.append(preds_dict.setdefault(user, ([], [])))

        scores_dict = {
            "target_items": true_items_for_user,
            "predicted_items": predicted_items,
            "item_embeddings": model.item_embedding.weight.cpu().detach(),
        }

    elif isinstance(model, BPR) or isinstance(model, LightGCN):
        logging.info(f"calculating scores for {model.__class__.__name__}")
        target_items = (
            pd.DataFrame(valid_data.dataset.inter_feat.interaction)[
                [config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]]
            ]
            .groupby(config["USER_ID_FIELD"])
            .agg(list)
            .reset_index()
        )
        targets_dict = dict(
            zip(
                target_items[config["USER_ID_FIELD"]],
                target_items[config["ITEM_ID_FIELD"]],
            )
        )
        users_with_scores = set(
            valid_data.dataset.inter_feat[config["USER_ID_FIELD"]].numpy()
        )
        true_items_for_user = [
            targets_dict[i] if i in users_with_scores else []
            for i in range(valid_data.dataset.user_num)
        ]
        with torch.no_grad():
            user_scores = [
                full_sort_scores([i], model.to("cpu"), valid_data).squeeze(0)
                if i in users_with_scores
                else []
                for i in range(valid_data.dataset.user_num)
            ]
        topk_user_scores = [
            list(x.topk(k=min(top_k_scores_cutoff, x.shape[0])))
            if len(x) > 0
            else ([], [])
            for x in user_scores
        ]

        scores_dict = {
            "target_items": true_items_for_user,
            "predicted_items": topk_user_scores,
            "item_embeddings": model.item_embedding.weight.cpu().detach(),
        }
    else:
        raise NotImplementedError

    logging.info(
        f"Saving true targets, user scores and items embeddings to {results_save_dir / 'scores.pt'}"
    )
    results_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        scores_dict,
        results_save_dir / "scores.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a recommendation model."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="ml-100k", help="Name of the dataset."
    )
    parser.add_argument(
        "--model_name", type=str, default="BPR", help="Name of the model to use."
    )
    parser.add_argument(
        "--top_k_scores_cutoff", type=int, default=2000, help="Top k scores cutoff."
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="configs",
        help="Directory containing configuration files.",
    )
    parser.add_argument(
        "--results_save_dir",
        type=str,
        default="training_results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        default=False,
        help="Force training from scratch.",
    )

    args = parser.parse_args()
    logging.info(f"Arguments: {args}")
    main(
        args.dataset_name,
        args.model_name,
        args.top_k_scores_cutoff,
        args.configs_dir,
        args.results_save_dir,
        args.force_train,
    )
