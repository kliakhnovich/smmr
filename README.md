# Sampled MMR
Repository for the Sampled MMR Research
This repo contains:
- production-ready code for experiments with SMMR in you projects/research
- implementaton of methods from paper (SMMR, MMR, DPP, SSD)
- code to reproduce results of experiments from paper
- citation info

# SMMR reranker usage

### Installation
~~~
cd production_ready
python3.11 -m venv smmr_env
source smmr_env/bin/activate
pip install -r requirements.txt
~~~

### Reranker configuration 
~~~ Python
reranker_params = {  
    "candidates_pool_size": 1000,  # Limits the scope of reranker to the top `candidates_pool_size` items with maximal relevances  
    "top_k": 200,                  # Size of reranker output  
    "lambda_": 0.99,               # Diversity tradeoff parameter  
    "scale_factor": 4,             # Batch size increasing rate  
    "temperature": 0.001,          # Controls the sharpness (likelihood of picking items from the end of the recommendations list)  
}  

from torch_rerankers import SampledMMRReranker  
reranker = SampledMMRReranker(**reranker_params)  
~~~

### Usage
You can find executable example in `example.py`
~~~ Python
# Works much faster if tensors are loaded to CUDA  
reranked_items, reranked_logits = reranker.rerank(  
    pred_logits=pred_logits,       # Logits of recommended items  
    label_ids=label_ids,           # Labels of recommended items  
    item_embeddings=item_embeddings,  # Embeddings of ALL UNIQUE items in the catalogue/batch (must be accessible by indexing over label_ids)  
)  
~~~

# To reproduce results:

### Step 0: Install dependancies
It is recommended to use python3.11
~~~
cd paper_reproducibility
python3.11 -m venv smmr_rep_env
source smmr_rep_env/bin/activate
pip install -r requirements.txt
~~~


### Step 1: Train model and get predicted relevance scores and embeddings
~~~
PYTHONPATH=. torchrun --nproc_per_node=1 src/cli/run_model_on_dataset.py <options>
~~~
#### Example
~~~
PYTHONPATH=. torchrun --nproc_per_node=1 src/cli/run_model_on_dataset.py \
    --dataset_name ml-100k \
    --model_name BPR \
    --results_save_dir training_results \
    --top_k_scores_cutoff 3000 
~~~


### Step 2: Run reranking algorithms with different hyperparameter sets
~~~
PYTHONPATH=. python src/cli/run_reranking_experiments.py <options>
~~~

#### Example

~~~
PYTHONPATH=. python src/cli/run_reranking_experiments.py \
    --dataset_name ml-100k \
    --model_name BPR \
    --scores_folder training_results \
    --reranking_results_dir reranking_results \
    --candidates_top_k 1000 \
    --rank_top_k 200 
~~~

# Citation

### If you find our method useful, please cite us
~~~
@inproceedings{10.1145/3726302.3730250,
author = {Liakhnovich, Kiryl and Lashinin, Oleg and Babkin, Andrei and Pechatov, Michael and Ananyeva, Marina},
title = {SMMR: Sampling-Based MMR Reranking for Faster, More Diverse, and Balanced Recommendations and Retrieval},
year = {2025},
isbn = {9798400715921},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3726302.3730250},
doi = {10.1145/3726302.3730250},
abstract = {Relevance and diversity are critical objectives in modern information retrieval (IR), particularly in recommender systems. Achieving a balance between relevance (exploitation) and diversity (exploration) optimizes user satisfaction and business goals such as catalog coverage and novelty. While existing post-processing reranking methods address this trade-off, they usually rely on greedy strategies, leading to suboptimal outcomes for large-scale tasks. To this end, we propose Sampled Maximal Marginal Relevance (SMMR), a novel sampling-based extension of MMR that introduces randomness into item selection to improve relevance-diversity trade-offs. SMMR avoids the rigidity of greedy and deterministic reranking, and achieves a logarithmic computational speedup, which allows it to scale on large candidate sets. Our evaluations on multiple real-world open-source datasets demonstrate that SMMR consistently outperforms existing state-of-the-art approaches, offering superior performance in balancing relevance and diversity. Our implementation of the proposed method is made available to support future research.},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2754–2758},
numpages = {5},
keywords = {candidates generation, diversity, information retrieval, post-processing methods, recommender systems, reranking},
location = {Padua, Italy},
series = {SIGIR '25}
}
~~~
