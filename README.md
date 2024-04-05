# Code for the ARR 2023 Submission - 'Order-Based Pre-training Strategies for Procedural Text Understanding'

## Required dependencies

Please run ```pip install -r requirements.txt```. 

## Hyperparameters for commands
- ```CONTEXT_STEPS:``` Number of steps to be used as the context. 
- ```EMBEDDING_TYPE:``` Type of embedding ('hamming', 'kenny', 'lehmer'). 
- ```MODEL_NAME:``` Name of the base transformer model ('bert', 'roberta') or a HuggingFace model which has the transformer model name as a substring ('contra-4-2_bert').
- ```MODEL_PATH:``` Path to the base transformer model ('bert-base-uncased', 'roberta-base'). 
- ```NUM_RANDOM_STEPS:``` Number of random steps to be sampled. 
- ```RECIPES_STEPS:``` Number of steps of recipes, or text about any process, in the training data. 
- ```TEST:``` Set 'True' to test code on your local machine (True, False). 
- ```TOTAL_NUM_PERMUTATIONS:``` Size of the permutation set. 

Before pre-training store your training data as a list of recipes with each recipe in turn as a list of steps in a pickle file called ```full_corpus.pickle```. Running the pretraining code for Permutation Classification or Embedding Regression should generate ```permutation_hamming_<RECIPES_STEPS>_<TOTAL_NUM_PERMUTATIONS>.pickle``` file which contains ```TOTAL_NUM_PERMUTATIONS``` permutations of length ```RECIPES_STEPS```. 

## Pretraining using Permutation Classification

Run ```python3 classification.py <MODEL_NAME> <MODEL_PATH> <RECIPES_STEPS> <TEST> <TOTAL_NUM_PERMUTATIONS>```. Running the command should generate a folder ```models/pc_<RECIPES_STEPS>_<TOTAL_NUM_PERMUTATIONS>__<MODEL_NAME>``` containing model and log files. Link to pretrained model [pc_6_100__roberta](https://huggingface.co/anony12sub34/pc_6_100__roberta). 

## Pretraining using Embedding Regression

Run ```python3 embedding.py <EMBEDDING_TYPE> <MODEL_NAME> <MODEL_PATH> <RECIPES_STEPS> <TEST> <TOTAL_NUM_PERMUTATIONS>```. Running the command should generate a folder ```models/embeddings_<EMBEDDING_TYPE>_<RECIPES_STEPS>_<TOTAL_NUM_PERMUTATIONS>__<MODEL_NAME>``` containing model and log files. 

## Pretraining using Skip-Clip

Run ```python3 skipclip.py <CONTEXT_STEPS> <MODEL_NAME> <MODEL_PATH> <NUM_RANDOM_STEPS> <TEST>```. Running the command should generate a folder ```models/skipclip_<CONTEXT_STEPS>_<NUM_RANDOM_STEPS>__<MODEL_NAME>``` containing model and log files. 

## Fine-tuning on SQuAD 2.0
- Pretrained models need to be finetuned on SQuAD 2.0 dataset before training on the entity-tracking downstream task. 
- To download the training set, run ```wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json```.
- Run ```python3 finetune_squad.py <MODEL_TYPE> <MODEL_PATH>```. 
- To get the models fine-tuned on SQuAD 2.0, follow the following format to get the link - ```https://huggingface.co/AnonymousSub/<SUBSTRING AFTER THE LAST '/' IN PRE-TRAINED MODEL LINK>_squad2.0```.

## Links to Pre-trained models  Fine-tuned on SQuAD 2.0

- Permutation Classification
1. [pc-6-50-roberta_squad2.0](https://huggingface.co/anony12sub34/pc-6-50-roberta_squad2.0)
2. [pc-6-200-roberta_squad2.0](https://huggingface.co/anony12sub34/pc-6-200-roberta_squad2.0)
- Embedding Regression
1. [embeddings-lehmer-6-50-roberta_squad2.0](https://huggingface.co/anony12sub34/embeddings-lehmer-6-50-roberta_squad2.0)
2. [embeddings-hamming-6-50-roberta_squad2.0](https://huggingface.co/anony12sub34/embeddings-hamming-6-50-roberta_squad2.0)
3. [embeddings-hamming-6-100-roberta_squad2.0](https://huggingface.co/anony12sub34/embeddings-hamming-6-100-roberta_squad2.0)
- Skip-Clip
1. [skipclip-4-4-roberta_squad2.0](https://huggingface.co/anony12sub34/skipclip-4-4-roberta_squad2.0)
