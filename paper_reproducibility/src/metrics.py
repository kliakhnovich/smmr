import numpy as np
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist


class Diversity_MeanInterList:
    """
    MeanInterList diversity measures the uniqueness of different users' recommendation lists.

    It can be used to measure how "diversified" are the recommendations different users receive.

    While the original proposal called this metric "Personalization", we do not use this name since the highest MeanInterList diversity
    is exhibited by a non personalized Random recommender.

    pag. 3, http://www.pnas.org/content/pnas/107/10/4511.full.pdf
    @article{zhou2010solving,
      title={Solving the apparent diversity-accuracy dilemma of recommender systems},
      author={Zhou, Tao and Kuscsik, Zolt{\'a}n and Liu, Jian-Guo and Medo, Mat{\'u}{\v{s}} and Wakeling, Joseph Rushton and Zhang, Yi-Cheng},
      journal={Proceedings of the National Academy of Sciences},
      volume={107},
      number={10},
      pages={4511--4515},
      year={2010},
      publisher={National Acad Sciences}
    }


    It was demonstrated by Ferrari Dacrema that this metric does not require to compute the common items all possible
    couples of users have in common but rather it is only sensitive to the total number of times each item has been recommended.
    MeanInterList diversity is a function of the square of the probability an item has been recommended to any user, hence
    MeanInterList diversity is equivalent to the Herfindahl index as they measure the same quantity.
    See
    @inproceedings{DBLP:conf/aaai/Dacrema21,
      author    = {Maurizio {Ferrari Dacrema}},
      title     = {Demonstrating the Equivalence of List Based and Aggregate Metrics
                   to Measure the Diversity of Recommendations (Student Abstract)},
      booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
                   2021, Thirty-Third Conference on Innovative Applications of Artificial
                   Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
                   in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
                   2021},
      pages     = {15779--15780},
      publisher = {{AAAI} Press},
      year      = {2021},
      url       = {https://ojs.aaai.org/index.php/AAAI/article/view/17886},
    }

    A TopPopular recommender that does not remove seen items will have 0.0 MeanInterList diversity.

    # The formula is diversity_cumulative += 1 - common_recommendations(user1, user2)/cutoff
    # for each couple of users, except the diagonal. It is VERY computationally expensive
    # We can move the 1 and cutoff outside of the summation. Remember to exclude the diagonal
    # co_counts = URM_predicted.dot(URM_predicted.T)
    # co_counts[np.arange(0, n_user, dtype=np.int):np.arange(0, n_user, dtype=np.int)] = 0
    # diversity = (n_user**2 - n_user) - co_counts.sum()/self.cutoff

    # If we represent the summation of co_counts separating it for each item, we will have:
    # co_counts.sum() = co_counts_item1.sum()  + co_counts_item2.sum() ...
    # If we know how many times an item has been recommended, co_counts_item1.sum() can be computed as how many couples of
    # users have item1 in common. If item1 has been recommended n times, the number of couples is n*(n-1)
    # Therefore we can compute co_counts.sum() value as:
    # np.sum(np.multiply(item-occurrence, item-occurrence-1))

    # The naive implementation URM_predicted.dot(URM_predicted.T) might require an hour of computation
    # The last implementation has a negligible computational time even for very big datasets

    """

    def __init__(self, n_items, cutoff):
        super(Diversity_MeanInterList, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float64)

        self.n_evaluated_users = 0
        self.n_items = n_items
        self.diversity = 0.0
        self.cutoff = cutoff

    def add_recommendations(self, recommended_items_ids):
        assert (
            len(recommended_items_ids) <= self.cutoff
        ), "Diversity_MeanInterList: recommended list is contains more elements than cutoff"

        self.n_evaluated_users += 1

        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):
        # Requires to compute the number of common elements for all couples of users
        if self.n_evaluated_users == 0:
            return 1.0

        cooccurrences_cumulative = (
            np.sum(self.recommended_counter**2) - self.n_evaluated_users * self.cutoff
        )

        # All user combinations except diagonal
        all_user_couples_count = self.n_evaluated_users**2 - self.n_evaluated_users

        diversity_cumulative = (
            all_user_couples_count - cooccurrences_cumulative / self.cutoff
        )

        self.diversity = diversity_cumulative / all_user_couples_count

        return self.diversity

    def get_theoretical_max(self):
        global_co_occurrence_count = (
            self.n_evaluated_users * self.cutoff
        ) ** 2 / self.n_items - self.n_evaluated_users * self.cutoff

        mild = 1 - 1 / (self.n_evaluated_users**2 - self.n_evaluated_users) * (
            global_co_occurrence_count / self.cutoff
        )

        return mild

    def merge_with_other(self, other_metric_object):
        assert (
            other_metric_object is Diversity_MeanInterList
        ), "Diversity_MeanInterList: attempting to merge with a metric object of different type"

        assert np.all(
            self.recommended_counter >= 0.0
        ), "Diversity_MeanInterList: self.recommended_counter contains negative counts"
        assert np.all(
            other_metric_object.recommended_counter >= 0.0
        ), "Diversity_MeanInterList: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter
        self.n_evaluated_users += other_metric_object.n_evaluated_users

    def get_diversity(preds: np.array, cutoff: int) -> dict[str, float]:
        diversity = Diversity_MeanInterList(preds.shape[1], cutoff)
        users_labels = np.argsort(-preds, axis=-1)[:, :cutoff]

        for user_labels in users_labels:
            diversity.add_recommendations(user_labels)

        diversity_value = diversity.get_metric_value()
        diversity_max = diversity.get_theoretical_max()

        return {"model_value": diversity_value, "max_value": diversity_max}


def precision(*, true_items, predicted_items, k, **kwargs):
    """
    Computes Precision@k for each user and returns the mean precision.

    Args:
        true_items (list of lists): Each sublist contains the true item indices for a user.
        predicted_items (2D array): (n_users, n_pred_items) array of predicted indices (top-k items).
        k (int): The cutoff value for k.

    Returns:
        float: Mean Precision@k across all users.
    """
    precisions = []
    for user_true, user_pred in zip(true_items, predicted_items):
        top_k_preds = set(user_pred[:k])
        num_hits = len(top_k_preds & set(user_true))
        precisions.append(num_hits / k)

    return np.mean(precisions)


def recall(*, true_items, predicted_items, k, **kwargs):
    """
    Computes Recall@k for each user and returns the mean recall.

    Args:
        true_items (list of lists): Each sublist contains the true item indices for a user.
        predicted_items (2D array): (n_users, n_pred_items) array of predicted indices (top-k items).
        k (int): The cutoff value for k.

    Returns:
        float: Mean Recall@k across all users.
    """
    recalls = []
    for user_true, user_pred in zip(true_items, predicted_items):
        top_k_preds = set(user_pred[:k])
        if len(user_true) == 0:
            recalls.append(0)
        else:
            num_hits = len(top_k_preds & set(user_true))
            recalls.append(num_hits / len(user_true))

    return np.mean(recalls)


def ndcg(*, true_items, predicted_items, k, **kwargs):
    """
    Computes NDCG@k for each user and returns the mean NDCG.

    Args:
        true_items (list of lists): Each sublist contains the true item indices for a user.
        predicted_items (2D array): (n_users, n_pred_items) array of predicted indices (top-k items).
        k (int): The cutoff value for k.

    Returns:
        float: Mean NDCG@k across all users.
    """

    def dcg_at_k(rel, k):
        """
        Compute DCG@k for relevance scores.
        """
        rel = np.array(rel[:k])
        if rel.size == 0:
            return 0
        return np.sum(rel / np.log2(np.arange(2, rel.size + 2)))

    def idcg_at_k(rel, k):
        """
        Compute ideal DCG@k (IDCG) for relevance scores.
        """
        rel = sorted(rel, reverse=True)
        return dcg_at_k(rel, k)

    ndcgs = []
    for user_true, user_pred in zip(true_items, predicted_items):
        rel = [1 if item in set(user_true) else 0 for item in user_pred[:k]]

        dcg = dcg_at_k(rel, k)
        idcg = idcg_at_k([1] * len(user_true), k)
        if idcg == 0:
            ndcgs.append(0)
        else:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs)


def item_coverage(*, predicted_items, k, total_items, **kwargs):
    """
    Computes Item Coverage@k.

    Args:
        predicted_items (2D array): (n_users, n_pred_items) array of predicted indices (ranked items per user).
        k (int): The cutoff value for k.
        total_items (int): Total number of items in the catalog.

    Returns:
        int: Item Coverage@k as a fraction of all unique items in recommendations.
    """
    recommended_items = set()
    for user_pred in predicted_items:
        recommended_items.update(user_pred[:k])

    return len(recommended_items) / total_items


def mean_inter_list_diversity(*, predicted_items, k, total_items, **kwargs):
    """
    Computes Diversity MeanInterList@k.

    Args:
        predicted_items (2D array): (n_users, n_pred_items) array of predicted indices (ranked items per user).
        k (int): The cutoff value for k.
        total_items (int): Total number of items in the catalog.

    Returns:
        dict: A dictionary containing:
            - "model_value": The Diversity MeanInterList value for the given recommendations.
            - "max_value": The theoretical maximum Diversity MeanInterList value for the given setup.
    """

    predicted_items = np.array(predicted_items)
    diversity = Diversity_MeanInterList(n_items=total_items, cutoff=k)

    for user_pred in predicted_items:
        diversity.add_recommendations(user_pred[:k])

    return diversity.get_metric_value()


def intra_list_diversity(*, predicted_items, k, item_embeddings, **kwargs):
    """
    Computes Intra-List Diversity@k for each user and returns the mean ILD.

    Args:
        predicted_items (2D array): (n_users, n_pred_items) array of predicted indices (ranked items per user).
        k (int): The cutoff value for k.
        item_embeddings (2D array): (n_items, embedding_dim) array of item embeddings.

    Returns:
        float: Mean Intra-List Diversity@k across all users.
    """
    il_diversities = []
    for user_pred in predicted_items:
        top_k_preds = user_pred[:k]
        if len(top_k_preds) <= 1:
            il_diversities.append(0.0)
            continue

        top_k_embeddings = item_embeddings[top_k_preds]

        cosine_sim_matrix = 1 - cdist(top_k_embeddings, top_k_embeddings, "cosine")

        upper_tri = np.triu(cosine_sim_matrix, k=1)

        pairwise_dissimilarities = upper_tri[upper_tri > 0]

        if pairwise_dissimilarities.size > 0:
            il_diversities.append(np.mean(pairwise_dissimilarities))
        else:
            il_diversities.append(0.0)

    return np.mean(il_diversities) if il_diversities else 0.0


def entropy(*, predicted_items, k, total_items, **kwargs):
    """
    Computes Entropy@k for recommendations.

    Args:
        predicted_items (2D array): (n_users, n_pred_items) - predicted indices of items per user.
        k (int): The cutoff value for k.
        total_items (int): Total number of items in the catalog.

    Returns:
        float: Entropy@k of the recommended items.
    """
    item_counts = Counter()
    for user_pred in predicted_items:
        top_k_preds = user_pred[:k]
        item_counts.update(top_k_preds)

    probabilities = np.zeros(total_items)
    total_recommendations = sum(item_counts.values())

    for item, count in item_counts.items():
        probabilities[item] = count / total_recommendations

    # Compute entropy: H = -Σ (p * log(p)) for non-zero probabilities
    entropy = -np.nansum(probabilities * np.log2(probabilities + 1e-9))

    return entropy


if __name__ == "__main__":
    true_items = [[1, 4, 3, 2, 5], [4, 5], [6, 7, 8, 9]]

    predicted_items = [[1, 4, 3, 2, 5], [4, 1, 5, 7, 3], [6, 7, 10, 8, 9]]

    k = 2

    precision_value = precision(
        true_items=true_items, predicted_items=predicted_items, k=k
    )
    recall_value = recall(true_items=true_items, predicted_items=predicted_items, k=k)
    ndcg_value = ndcg(true_items=true_items, predicted_items=predicted_items, k=k)

    print(f"Precision@{k}: {precision_value:.4f}")
    print(f"Recall@{k}: {recall_value:.4f}")
    print(f"NDCG@{k}: {ndcg_value:.4f}")

    true_items = [[3, 5, 7], [2, 6], [1, 4, 9]]
    predicted_items = [
        [3, 1, 7, 9, 10],
        [6, 2, 8, 5, 11],
        [4, 9, 1, 7, 12],
    ]
    _item_embeddings = np.random.rand(20, 50)
    k = 3
    total_items = 13

    recall_value = recall(
        true_items=true_items,
        predicted_items=predicted_items,
        k=k,
        total_items=total_items,
    )
    item_coverage_value = item_coverage(
        predicted_items=predicted_items, k=k, total_items=total_items
    )
    mean_inter_list_diversity_value = mean_inter_list_diversity(
        predicted_items=predicted_items, k=k, total_items=total_items
    )
    intra_list_diversity_value = intra_list_diversity(
        predicted_items=predicted_items, k=k, item_embeddings=_item_embeddings
    )
    entropy_value = entropy(
        predicted_items=predicted_items, k=k, total_items=total_items
    )

    print(f"Recall@{k}: {recall_value}")
    print(f"Item Coverage@{k}: {item_coverage_value}")
    print(f"Mean Inter-List Diversity@{k}: {mean_inter_list_diversity_value}")
    print(f"Intra-List Diversity@{k}: {intra_list_diversity_value}")
    print(f"Entropy@{k}: {entropy_value}")
