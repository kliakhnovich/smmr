from torch_rerankers import SampledMMRReranker
import torch



if __name__ == "__main__":

    reranker_params = {
        "candidates_pool_size": 1000,
        "top_k": 200,
        "lambda_": 0.99,
        "scale_factor": 4,
        "temperature": 0.001,
    }
    reranker = SampledMMRReranker(**reranker_params)

    ### EXAMPLE DATA GENERATION ###

    # generate random data for test
    n_unique_items = 50_000
    batch_size = 32
    item_embedding_dim = 256
    recommendations_per_user = 3000

    # generate list of random labels of shape (batch_size, recommendations_per_user)
    user_recommendations_labels = torch.vstack([
        torch.randperm(n_unique_items)[:recommendations_per_user]
        for _ in range(batch_size)
    ])

    user_recommendations_relevances_logits = torch.rand(
        batch_size, recommendations_per_user)

    # generate random item embeddings matrix of shape (n_unique_items, item_embedding_dim)
    item_embeddings = torch.rand(n_unique_items, item_embedding_dim)


    print(f'\n{user_recommendations_relevances_logits.shape=}')
    print(f'{user_recommendations_labels.shape=}')
    print(f'{item_embeddings.shape=}')

    ### Reranker Usage ###

    reranked_items, reranked_logits = reranker.rerank(
        pred_logits=user_recommendations_relevances_logits,
        label_ids=user_recommendations_labels,
        item_embeddings=item_embeddings,
    )

    print('\nreranking results:')
    print(f'{reranked_items.shape=}')
    print(f'{reranked_logits.shape=}')
    print('\nsome logits:')
    print(reranked_logits[:3, :8])


