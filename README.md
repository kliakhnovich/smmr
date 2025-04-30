# Sampled MMR
Repository for the Sampled MMR Research
This repo contains:
- implementaton of methods from paper (SMMR, MMR, DPP, SSD)
- code to reproduce results of experiments from paper


to reproduce results:

## Step 0: Install dependancies
~~~
python -m pip install -r requirements.txt
~~~



## Step 1: Train model and get predicted relevance scores and embeddings
~~~
PYTHONPATH=. torchrun --nproc_per_node=1 src/cli/run_model_on_dataset.py <options>
~~~
### Example
~~~
PYTHONPATH=. torchrun --nproc_per_node=1 src/cli/run_model_on_dataset.py \
    --dataset_name ml-100k \
    --model_name BPR \
    --results_save_dir training_results \
    --top_k_scores_cutoff 3000 
~~~


## Step 2: Run reranking algorithms with different hyperparameter sets
~~~
PYTHONPATH=. python src/cli/run_reranking_experiments.py <options>
~~~

### Example

~~~
PYTHONPATH=. python src/cli/run_reranking_experiments.py \
    --dataset_name ml-100k \
    --model_name BPR \
    --scores_folder training_results \
    --reranking_results_dir reranking_results \
    --candidates_top_k 1000 \
    --rank_top_k 200 
~~~
