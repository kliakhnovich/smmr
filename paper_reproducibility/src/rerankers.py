import math
import numpy as np
from src.utils import multiply_non_diagonal, temperatured_softmax


# is just a top-k
def no_rerank(pred_logits, label_ids, item_embeddings, top_k):
    top_k_ids = np.argsort(pred_logits)[::-1][:top_k]
    return label_ids[top_k_ids]


def dpp(kernel_matrix, max_length, epsilon=1e-10):
    """
    source:
    https://github.com/laming-chen/fast-map-dpp/blob/6ab745ca1273941412a7f32ce6432549cfe4b5af/dpp.py#L5

    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_rerank_user(
    pred_logits, label_ids, item_embeddings, top_k, alpha, epsilon=1e-6
):
    relevance_scores = 1 / (1 + np.exp(-pred_logits))
    candidate_embeddings = item_embeddings[label_ids]

    assert alpha <= 1.0
    assert top_k <= len(relevance_scores)
    normalized_embeddings = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    quality_scores = np.outer(relevance_scores, relevance_scores)
    kernel_matrix = similarity_matrix * quality_scores
    kernel_matrix = multiply_non_diagonal(kernel_matrix, alpha)

    selected_indices = dpp(kernel_matrix, top_k, epsilon)

    return label_ids[selected_indices]


def mmr_rerank_user(pred_logits, label_ids, item_embeddings, top_k, lambda_):
    relevance_scores = 1 / (1 + np.exp(-pred_logits))
    candidate_embeddings = item_embeddings[label_ids]

    normalized_emb = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, ord=2, axis=1, keepdims=True
    )
    similarity_matrix = normalized_emb @ normalized_emb.T

    selected_indices = [relevance_scores.argmax()]
    candidate_indices = list(range(len(label_ids)))
    candidate_indices.remove(selected_indices[0])

    for _ in range(top_k - 1):
        submatrix = similarity_matrix[np.ix_(selected_indices, candidate_indices)]
        max_similarity_scores = np.max(submatrix, axis=0)

        mmr_score = (
            lambda_ * relevance_scores[candidate_indices]
            - (1 - lambda_) * max_similarity_scores
        )
        id_to_select = candidate_indices[mmr_score.argmax()]
        candidate_indices.remove(id_to_select)
        selected_indices.append(id_to_select)

    return label_ids[selected_indices]


def ssd_rerank_user(pred_logits, label_ids, item_embeddings, top_k, gamma):
    """
    source:
    Yanhua Huang, Weikun Wang, Lei Zhang, Ruiwen Xu. 2021. Sliding Spectrum Decomposition for Diversified Recommendation
    https://arxiv.org/pdf/2107.05204
    """

    relevance_scores = 1 / (1 + np.exp(-pred_logits))
    candidate_embeddings = item_embeddings[label_ids]
    candidate_embeddings = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, ord=2, axis=1, keepdims=True
    )

    last_selected_idx = relevance_scores.argmax()
    selected_indices = [last_selected_idx]
    candidate_indices = list(range(len(label_ids)))
    candidate_indices.remove(last_selected_idx)

    t = 1
    V = gamma * np.linalg.norm(candidate_embeddings[last_selected_idx], ord=2)

    while t < top_k:
        last_selected_embedding = candidate_embeddings[last_selected_idx]
        dot_products = candidate_embeddings[candidate_indices] @ last_selected_embedding
        projection_norm = last_selected_embedding @ last_selected_embedding
        projections = np.outer(dot_products / projection_norm, last_selected_embedding)
        candidate_embeddings[candidate_indices] -= projections

        ssd_scores = (
            relevance_scores[candidate_indices]
            + np.linalg.norm(candidate_embeddings[candidate_indices], axis=1) * V
        )

        last_selected_idx = candidate_indices[np.argmax(ssd_scores)]
        selected_indices.append(last_selected_idx)
        t += 1
        candidate_indices.remove(last_selected_idx)
        V = V * np.linalg.norm(candidate_embeddings[last_selected_idx])

    return label_ids[selected_indices]


def sampled_mmr_rerank_user(
    pred_logits,
    label_ids,
    item_embeddings,
    top_k,
    lambda_,
    scale_factor,
    temperature,
):
    """
    Description:
    - Sampled MMR reranking algorithm for top-k recommendation.

    args:
    pred_logits: [batch_size, num_items]
    label_ids: [batch_size, num_items]
    item_embeddings: [num_items, embedding_dim]
    top_k: int
    lambda_: float
    scale_factor: float
    temperature: float

    """
    relevance_scores = 1 / (1 + np.exp(-pred_logits))
    candidate_embeddings = item_embeddings[label_ids]

    assert top_k <= len(relevance_scores)
    assert scale_factor >= 1
    assert temperature >= 0.0
    assert (lambda_ >= 0.0) and (lambda_ <= 1)

    normalized_emb = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, ord=2, axis=1, keepdims=True
    )
    similarity_matrix = normalized_emb @ normalized_emb.T

    selected_indices = [relevance_scores.argmax()]
    candidate_indices = list(range(len(label_ids)))
    candidate_indices.remove(selected_indices[0])

    while len(selected_indices) < top_k:
        submatrix = similarity_matrix[np.ix_(selected_indices, candidate_indices)]

        max_similarity_scores = np.max(submatrix, axis=0)

        mmr_score = (
            lambda_ * relevance_scores[candidate_indices]
            - (1 - lambda_) * max_similarity_scores
        )
        mmr_probs = temperatured_softmax(mmr_score, temperature)
        sample_size = max(
            1,
            min(
                np.count_nonzero(mmr_probs),
                int(len(selected_indices) * (scale_factor - 1)),
            ),
        )
        samples_without_replacement = np.random.choice(
            candidate_indices, size=sample_size, replace=False, p=mmr_probs
        )
        selected_indices.extend(samples_without_replacement)
        candidate_indices = list(
            set(candidate_indices) - set(samples_without_replacement)
        )

    selected_indices = selected_indices[:top_k]

    return label_ids[selected_indices]
