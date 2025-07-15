# Sampled MMR
Repository for the Sampled MMR Research
This repo contains:
- production-ready code for experiments with SMMR in you projects/research
- implementaton of methods from paper (SMMR, MMR, DPP, SSD)
- code to reproduce results of experiments from paper

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
    "candidates_pool_size": 1000, # limits the scope of reranker on candidates_pool_size items with maximal relevances
    "top_k": 200, # size of reranker output
    "lambda_": 0.99, # diversity tradeoff parameter
    "scale_factor": 4, # batch size increasing rate
    "temperature": 0.001, # controls the sharpness (how likely it is to pick item from the end of recommendations list )
}

from torch_rerankers import SampledMMRReranker
reranker = SampledMMRReranker(**reranker_params)
~~~

### Usage
You can find executable example in `example.py`
~~~ Python
# works much faster if tensors are loaded to CUDA
reranked_items, reranked_logits = reranker.rerank(
    pred_logits=pred_logits, # logits of recommended items
    label_ids=label_ids, # labels of recommended items
    item_embeddings=item_embeddings, # embeddings of ALL UNIQUE items in catalogue
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
