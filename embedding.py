from itertools import combinations
from math import factorial, sqrt
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments 
from torch.nn import MSELoss 
from utils import get_training_data

import numpy as np
import os
import pickle
import random
import sys
import torch
import wandb

# os.environ['WANDB_API_KEY'] = ''
# os.environ["WANDB_MODE"] = 'dryrun' # creates a local cache of the run, including loss and learning rate at every batch

print(f"sys.argv: {sys.argv}")

EMBEDDING_TYPE = sys.argv[1] # ['hamming', 'kenny', 'lehmer']
MODEL_NAME = sys.argv[2] # ['bert', 'roberta']
MODEL_PATH = sys.argv[3] # ['bert-base-uncased', 'roberta-base']
RECIPES_STEPS = int(sys.argv[4]) # [6, 9]
TEST = (sys.argv[5]=='True')
TOTAL_NUM_PERMUTATIONS = int(sys.argv[6]) # [2, 10, 50, 100, 200, 1000] 

HAMMING_SELECTION = 'max' # (max) Sample selected per iteration based on hamming distance: [max] highest; [mean] average 
NUM_DATA = int(1e6) # (1e6)
PERMUTATION_HEURISTIC = 'hamming' # (random) heuristic to select permutations ['random', 'hamming'] 
RECIPES_FILE_PATH = 'full_corpus.pickle' # ['first_100_recipes', 'full_corpus.pickle']
SEED = 42 # (42)

# training 
# https://discuss.huggingface.co/t/colab-ram-crash-error-fine-tuning-roberta-in-colab/2830/2
LOGGING_STEPS = 500 
SAVE_STRATEGY = 'epoch'
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 1 
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

if TEST:
    print('TESTING')
    LOGGING_STEPS = 1
    NUM_DATA = 20
    RECIPES_FILE_PATH = 'full_corpus.pickle'
    SAVE_STRATEGY = 'no'
    TRAIN_BATCH_SIZE = 8

assert factorial(RECIPES_STEPS)>=TOTAL_NUM_PERMUTATIONS, f"TOTAL_NUM_PERMUTATIONS:{TOTAL_NUM_PERMUTATIONS} is more than factorial(RECIPES_STEPS): {factorial(RECIPES_STEPS)}"
    
if 'bert' in MODEL_NAME:
    transformer_model = 'bert-base-uncased'
if 'roberta' in MODEL_NAME:
    transformer_model = 'roberta-base' 
if 'distilbert' in MODEL_NAME:
    transformer_model = 'distilbert-base-uncased'

model_name = f"embeddings_{EMBEDDING_TYPE}_{RECIPES_STEPS}_{TOTAL_NUM_PERMUTATIONS}__{MODEL_NAME}"
if TEST:
    model_name = f'embeddings_{EMBEDDING_TYPE}_test'
    os.environ['WANDB_DISABLED'] = 'true'

model_dir_path = Path(f'models/{model_name}')
print(f"model_dir_path: {model_dir_path}")
if not TEST:
    wandb.init(project='embeddings', name=model_name)
    assert not model_dir_path.exists() or (model_dir_path.exists() and len(list(model_dir_path.iterdir()))==0), f"{str(model_dir_path)} already exists"
    model_dir_path.mkdir(parents=True, exist_ok=True)


class embeddingsTrainer(Trainer):
    # TODO: figure out how to pass codes as a key in input
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        criterion = MSELoss()
        loss = criterion(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
        

class RecipesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def KemenyEmbed_no_offset_unisign(sigma):
    n=len(sigma)
    phi=np.zeros(int((n*(n-1)/2)))
    count=0
    for (i,j) in combinations(range(n),2):
        phi[count]=int(np.sign(sigma[j]-sigma[i])>0)
        count+=1
    return phi

def HammingEmbed(sigma):    
    phi=np.zeros((len(sigma),len(sigma)))
    for i in sigma:
        j=sigma[i]
        phi[i-1,j]+=1
    return phi.reshape(len(sigma)**2)


hp = {
    'HAMMING_SELECTION': HAMMING_SELECTION,
    'NUM_DATA': NUM_DATA,
    'PERMUTATION_HEURISTIC': PERMUTATION_HEURISTIC,
    'RECIPES_STEPS': RECIPES_STEPS,
    'RECIPES_FILE_PATH': RECIPES_FILE_PATH,
    'SEED': SEED,
    'TOTAL_NUM_PERMUTATIONS': TOTAL_NUM_PERMUTATIONS,
}
recipes_documents, labels, permutation_list = get_training_data(hp)

if EMBEDDING_TYPE=='hamming':
    print("preparing hamming embedding")
    labels = [HammingEmbed(permutation_list[label]) for label in labels]
    num_labels = RECIPES_STEPS*RECIPES_STEPS 
elif EMBEDDING_TYPE=='kenny': 
    print("preparing kenny embedding")
    labels = [KemenyEmbed_no_offset_unisign(permutation_list[label]) for label in labels]
    num_labels = int(RECIPES_STEPS*(RECIPES_STEPS-1)/2)
elif EMBEDDING_TYPE=='lehmer':
    print("preparing lehmer embedding")
    labels = [[max(i-w,0) for i,w in enumerate(permutation_list[label])] for label in labels]
    num_labels = RECIPES_STEPS
else:
    assert f"{EMBEDDING_TYPE} is not a valid embedding type"


tokenizer = AutoTokenizer.from_pretrained(transformer_model)

train_encodings = tokenizer(recipes_documents, truncation=True, padding=True)
train_dataset = RecipesDataset(train_encodings, labels)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels) # getting IndexError without this kwarg 

training_args = TrainingArguments(
    output_dir=str(model_dir_path.joinpath('results')),          # output directory
    num_train_epochs=TRAIN_EPOCHS,              # total number of training epochs
    per_device_train_batch_size=TRAIN_BATCH_SIZE,  # batch size per device during training
    warmup_steps=WARMUP_STEPS,                # number of warmup steps for learning rate scheduler
    weight_decay=WEIGHT_DECAY,               # strength of weight decay
    logging_dir=str(model_dir_path.joinpath('logs')),            # directory for storing logs
    logging_steps=LOGGING_STEPS,
    save_strategy=SAVE_STRATEGY,
    seed=SEED, 
)

trainer = embeddingsTrainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    tokenizer=tokenizer, 
)

trainer.train()
