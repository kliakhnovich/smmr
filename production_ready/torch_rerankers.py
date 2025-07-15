import inspect
import json
import typing as t
from abc import abstractmethod

import torch
from autoregistry import Registry
from cyclopts import convert
from torchtyping import TensorType

__all__ = [
    "Reranker",
    "NoRerank",
    "SampledMMRReranker",
]


class Reranker(Registry, case_sensitive=False, snake_case=True):
    name: str
    _parameter_to_annotation: dict[str, type] = {}

    def __init__(
        self,
        *,
        candidates_pool_size: int,
        top_k: int,
    ) -> None:
        if top_k > candidates_pool_size:
            raise ValueError(
                f"top_k ({top_k }) should be less than candidates_pool_size ({candidates_pool_size})",
            )

        self.candidates_pool_size = candidates_pool_size
        self.top_k = top_k

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = cls.__registry__.name

        for parameter in inspect.signature(cls.__init__).parameters.values():
            if (name := parameter.name) in ["self"]:
                continue
            if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(f"{name} must be keyword only argument")
            if (annotation := parameter.annotation) == inspect.Parameter.empty:
                raise TypeError(f"{name} must have type-annotation")

            cls._parameter_to_annotation[name] = annotation

    @classmethod
    def from_json_string(
        cls,
        parameter_string_json: str,
        /,
        top_k: int | None = None,
    ) -> t.Self:
        parameters = json.loads(
            parameter_string_json,
            # hook to keep all values in str (for compatibility with cyclopts.convert)
            object_hook=lambda d: {k: str(v) for k, v in d.items()},
        )

        kwargs = {}
        for parameter, value in parameters.items():
            if (annotation := cls._parameter_to_annotation.get(parameter)) is None:
                raise ValueError(f"got unknown parameter {parameter} for reranker {cls.name}")
            try:
                kwargs[parameter] = convert(annotation, [value])
            except BaseException as err:
                raise TypeError(
                    f"cannot convert {parameter} parameter value {value} to expected type {annotation} "
                    f"for reranker {cls.name}",
                ) from err

        if top_k is not None:
            kwargs["top_k"] = top_k

        return cls(**kwargs)

    @abstractmethod
    def _rerank(
        self: t.Self,
        batch_pred_logits: TensorType["batch_size", "num_candidates", float],  # noqa: F821
        batch_label_ids: TensorType["batch_size", "num_candidates", int],  # noqa: F821
        item_embeddings: TensorType["num_all_unique_items", "embedding_size", float],  # noqa: F821
    ) -> tuple[TensorType["batch_size", "num_candidates", int], TensorType["batch_size", "num_candidates", float]]:  # noqa: F821
        # has to be implemented in child classes
        raise NotImplementedError

    def rerank(
        self: t.Self,
        pred_logits: TensorType["batch_size", "num_candidates", float],  # noqa: F821,
        item_embeddings: TensorType["num_all_unique_items", "embedding_size", float] = None,  # noqa: F821,
        label_ids: TensorType["batch_size", "num_candidates", int] = None,  # noqa: F821,
    ) -> tuple[TensorType["batch_size", "num_candidates", int], TensorType["batch_size", "num_candidates", float]]:  # noqa: F821
        if self.candidates_pool_size > pred_logits.shape[1]:
            raise ValueError(
                f"candidates_pool_size ({self.candidates_pool_size}) should not be greater than pred_logits.shape[1] ({pred_logits.shape[1]})",
            )

        pred_logits, top_k_idx = pred_logits.topk(self.candidates_pool_size, dim=1)
        label_ids = (
            label_ids[torch.arange(top_k_idx.shape[0]).unsqueeze(1), top_k_idx] if label_ids is not None else top_k_idx
        )

        reranked_items, reranked_logits = self._rerank(
            pred_logits,
            label_ids,
            item_embeddings,
        )

        return reranked_items, reranked_logits

    def __call__(self, *args, **kwargs) -> tuple[TensorType["batch_size", "num_candidates", int], TensorType["batch_size", "num_candidates", float]]:  # noqa: F821:
        return self.rerank(*args, **kwargs)


class NoRerank(Reranker):
    def __init__(
        self,
        *,
        candidates_pool_size: int,
        top_k: int,
    ) -> None:
        super().__init__(
            candidates_pool_size=candidates_pool_size,
            top_k=top_k,
        )

    def _rerank(
        self: t.Self,
        batch_pred_logits: TensorType["batch_size", "num_candidates", float],  # noqa: F821
        batch_label_ids: TensorType["batch_size", "num_candidates", int],  # noqa: F821
        item_embeddings: TensorType["num_all_unique_items", "embedding_size", float] = None,  # noqa: F821
    ) -> tuple[TensorType["batch_size", "num_candidates", int], TensorType["batch_size", "num_candidates", float]]:  # noqa: F821
        batch_reranked_logits, batch_top_k_idxs = batch_pred_logits.topk(self.top_k, dim=1)
        batch_reranked_label_ids = batch_label_ids[
            torch.arange(batch_top_k_idxs.shape[0]).unsqueeze(1),
            batch_top_k_idxs,
        ]
        return batch_reranked_label_ids, batch_reranked_logits


class SampledMMRReranker(Reranker):
    ZERO_PROB_EPSILON = 1e-30

    def __init__(
        self,
        *,
        candidates_pool_size: int,
        top_k: int,
        lambda_: float,
        scale_factor: float,
        temperature: float,
    ) -> None:
        super().__init__(
            candidates_pool_size=candidates_pool_size,
            top_k=top_k,
        )

        if scale_factor <= 1:
            raise ValueError(f"scale_factor ({scale_factor}) should be >= 1")
        if temperature < 0.0:
            raise ValueError(f"temperature ({temperature}) should be >= 0")

        self.lambda_ = lambda_
        self.scale_factor = scale_factor
        self.temperature = temperature

    @staticmethod
    def temperatured_softmax(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x = x - torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp(x / temperature)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

    def _rerank(
        self: t.Self,
        batch_pred_logits: TensorType["batch_size", "num_candidates", float],  # noqa: F821
        batch_label_ids: TensorType["batch_size", "num_candidates", int],  # noqa: F821
        item_embeddings: TensorType["num_all_unique_items", "embedding_size", float],  # noqa: F821
    ) -> tuple[TensorType["batch_size", "num_candidates", int], TensorType["batch_size", "num_candidates", float]]:  # noqa: F821
        batch_relevance_scores = 1 / (1 + torch.exp(-batch_pred_logits))
        batch_candidate_embeddings = item_embeddings[batch_label_ids]
        batch_selected_mask = torch.full_like(batch_relevance_scores, False, dtype=torch.bool)

        batch_normalized_emb = batch_candidate_embeddings / batch_candidate_embeddings.norm(p=2, dim=-1, keepdim=True)
        batch_similarity_matrix = batch_normalized_emb @ batch_normalized_emb.permute(0, 2, 1)

        batch_selected_indices = batch_relevance_scores.argmax(dim=1, keepdim=True)
        batch_selected_mask[torch.arange(batch_selected_mask.size(0)), batch_selected_indices.squeeze()] = True

        while batch_selected_indices.shape[1] < self.top_k:
            batch_submatrices = batch_similarity_matrix[batch_selected_mask].view(
                batch_selected_mask.shape[0],
                -1,
                batch_selected_mask.shape[1],
            )
            batch_max_similarity_scores = batch_submatrices.max(dim=1).values  # noqa: PD011
            batch_mmr_score = self.lambda_ * batch_relevance_scores - (1 - self.lambda_) * batch_max_similarity_scores

            # make already selected items imposible to sample via masking
            batch_mmr_score[batch_selected_mask] = -torch.inf
            batch_mmr_probs = self.temperatured_softmax(
                batch_mmr_score,
                self.temperature,
            )
            batch_mmr_probs += (~batch_selected_mask) * self.ZERO_PROB_EPSILON

            sample_size = max(
                1,
                min(
                    int(batch_selected_indices.shape[1] * (self.scale_factor - 1)),
                    (batch_relevance_scores.shape[1] - batch_selected_indices.shape[1]),
                ),
            )
            batch_sampled_indices = torch.multinomial(batch_mmr_probs, sample_size, replacement=False)
            batch_selected_indices = torch.cat((batch_selected_indices, batch_sampled_indices), dim=-1)
            batch_selected_mask[torch.arange(batch_selected_mask.size(0)).unsqueeze(1), batch_sampled_indices] = True

        batch_reranked_label_ids = batch_label_ids[
            torch.arange(batch_label_ids.shape[0]).unsqueeze(1),
            batch_selected_indices[:, : self.top_k],
        ]

        batch_reranked_logits = batch_pred_logits[
            torch.arange(batch_pred_logits.shape[0]).unsqueeze(1),
            batch_selected_indices[:, : self.top_k],
        ]

        return batch_reranked_label_ids, batch_reranked_logits
