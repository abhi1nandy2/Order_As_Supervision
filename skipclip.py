import os
import sys
import pickle
import numpy as np
import random as rn
import torch

import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig 
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)

from functools import partial
from tqdm import tqdm
import math
import wandb

if os.path.exists('model_cache'):
    os.system('rm -r model_cache')

# os.environ['WANDB_API_KEY'] = ''
# os.environ['WANDB_MODE'] = 'dryrun'

print(f"sys.argv: {sys.argv}")

CONTEXT_STEPS = int(sys.argv[1]) # [4]
MODEL_NAME = sys.argv[2] # ['bert']
MODEL_PATH = sys.argv[3] # ['bert-base-uncased']
NUM_RANDOM_STEPS = int(sys.argv[4]) # [4]
TEST = (sys.argv[5]=='True') # [False, True]

MAX_CONTEXT_LEN = 128
MAX_STEP_LEN = 32
MIN_STEPS_PER_RECIPE = CONTEXT_STEPS + NUM_RANDOM_STEPS
NUM_DATA = int(1e6) 
SEED = 0
USE_CONTRASTIVE = False 

# training 
ACCUM_ITER = 2
BATCH_SIZE = 32
EPOCHS = 1
EPS = 1e-8
LOG_EVERY = 250
LR = 5e-5 
MARGIN = 0.2
NUM_WORKERS = 8
WARMUP_STEPS = 0

recipe_corpus_filepath = 'full_corpus.pickle'

# model_name_suffix = f'{CONTEXT_STEPS}_{MIN_STEPS_PER_RECIPE}_{NUM_DATA}_{NUM_RANDOM_STEPS}_{SEED}_{USE_CONTRASTIVE}'
model_name_suffix = f'{CONTEXT_STEPS}_{NUM_RANDOM_STEPS}'
save_model_name = 'skipclip_' + model_name_suffix + f'__{MODEL_NAME}' 

if TEST:
    BATCH_SIZE = 2
    EPOCHS = 1
    LOG_EVERY = 1
    NUM_DATA = 4
    NUM_RANDOM_STEPS = 2
    recipe_corpus_filepath = 'full_corpus.pickle'
    save_model_name = 'skipclip_test'

project_name = save_model_name
print(f"project_name: {project_name}")

save_model_folder = 'models/' + save_model_name + '/'

from pathlib import Path
save_model_folder_path = Path(save_model_folder)
if save_model_folder_path.exists() and len(list(save_model_folder_path.iterdir()))>0 and not TEST:
    print(f"{save_model_folder} already exists")
else:
    save_model_folder_path.mkdir(exist_ok=True,parents=True)

print(f"save_model_name: {save_model_name}")

rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

with open(recipe_corpus_filepath, 'rb') as fr:
    corpus = pickle.load(fr)

corpus = [item for item in corpus if len(item)>=MIN_STEPS_PER_RECIPE]

print('Number of recipes >= {} = {}'.format(MIN_STEPS_PER_RECIPE, len(corpus)))
print()

corpus = rn.sample(corpus, min(len(corpus),NUM_DATA))
if NUM_DATA>len(corpus):
    corpus += [rn.choice(corpus) for _ in range(NUM_DATA-len(corpus))]
print('Number of sampled recipes = {}'.format(len(corpus)))
print()

# https://huggingface.co/course/chapter6/3
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir = './model_cache', use_fast=True)
print(f"tokenizer.is_fast: {tokenizer.is_fast}")


class RecipesDataset(torch.utils.data.Dataset):
    def __init__(self, context_encodings, step_encodings_list, labels, use_contrastive, neg_step_encodings_list=[]):
        self.context_encodings = encodings
        self.step_encodings_list = step_encodings_list
        self.labels = labels
        self.use_contrastive = use_contrastive
        self.neg_step_encodings_list = neg_step_encodings_list

    def __getitem__(self, idx):
        item['context_encodings'] = {key: torch.tensor(val[idx]) for key, val in self.context_encodings.items()}
        item['step_encodings_list'] = [{key: torch.tensor(val[idx]) for key, val in step_encodings.items()} for step_encodings in step_encodings_list]
        item['labels'] = torch.tensor(self.labels[idx])
        if self.use_contrastive:
            item['neg_step_encodings_list'] = [{key: torch.tensor(val[idx]) for key, val in neg_step_encodings.items()} for neg_step_encodings in neg_step_encodings_list]
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_function(examples, max_length):
    return tokenizer(examples, max_length = max_length, padding = 'max_length', truncation = True)

print('tokenizing recipes')
# https://stackoverflow.com/questions/13499824/using-map-function-with-keyword-arguments
context_tokenize_function = partial(tokenize_function, max_length=CONTEXT_STEPS)
context_encodings = list(map(context_tokenize_function, corpus))

print('tokenizing steps')
random_indices_list_list = [sorted(rn.sample(np.arange(CONTEXT_STEPS, len(recipe)).tolist(), NUM_RANDOM_STEPS)) for recipe in corpus]
step_tokenize_function = partial(tokenize_function, max_length=MAX_STEP_LEN)
step_encodings_list = [list(map(step_tokenize_function,[recipe[random_indices_list[i]] for random_indices_list, recipe in zip(random_indices_list_list, corpus)])) for i in range(NUM_RANDOM_STEPS)]

neg_step_encodings_list = []
if USE_CONTRASTIVE:
    print('tokenizing negetive steps')
    recipe_indices_list = [rn.choice(np.arange(0, i).tolist() + np.arange(i+1, len(corpus)).tolist()) for i in range(len(corpus))]
    random_indices_list_list = [sorted(rn.sample(range(len(corpus[recipe_index])), NUM_RANDOM_STEPS)) for recipe_index in recipe_indices_list]
    neg_step_encodings_list = [list(map(step_tokenize_function, [corpus[recipe_index][random_indices_list[i]] for random_indices_list, recipe_index in \
                                                                 zip(random_indices_list_list, recipe_indices_list)])) for i in range(NUM_RANDOM_STEPS)]

train_dataset = RecipesDataset(context_encodings, step_encodings_list, labels, USE_CONTRASTIVE, neg_step_encodings_list=neg_step_encodings_list)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
len_dataloader = len(train_dataloader)
total_steps = len(train_dataloader) * EPOCHS
print()
print('Loaded all data')
print()

class RankUsingPreviousContext(nn.Module):

    def __init__(self, model_path, num_random_steps, use_contrastive):
        super(RankUsingPreviousContext, self).__init__()
        self.model_layer = AutoModel.from_pretrained(model_path)
        self.num_random_steps = self.num_random_steps
        self.use_contrastive = self.use_contrastive
        
        self.cos = nn.CosineSimilarity(dim=-1)    

    def forward(self, context_encodings, step_encodings_list, neg_step_encodings_list):

        context_output = self.model_layer.forward(context_encodings).last_hidden_state[:, 0, :]

        score_list = []
        for i in range(self.num_random_steps):
            avg_output = self.model_layer.forward(step_encodings_list[i]).last_hidden_state[:, 0, :]
            score_list.append(self.cos(context_output, avg_output).reshape(-1))

        if self.use_contrastive:
            for i in range(self.num_random_steps):
                neg_avg_output = self.model_layer.forward(neg_step_encodings_list[i]).last_hidden_state[:, 0, :]
                score_list.append(self.cos(context_output, neg_avg_output).reshape(-1))

        return score_list


model = nn.DataParallel(RankUsingPreviousContext(model_path=MODEL_PATH, num_random_steps=NUM_RANDOM_STEPS, use_contrastive=USE_CONTRASTIVE))
model.to(device)

wandb.init(project='rupc', name=project_name)

criterion = nn.MarginRankingLoss(margin=MARGIN)
dummy_criterion = nn.MarginRankingLoss()

optimizer = AdamW(model.parameters(),
                  lr = LR, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = EPS # args.adam_epsilon  - default is 1e-8.
                )

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = WARMUP_STEPS, # Default value in run_glue.py
                                            num_training_steps = math.ceil(total_steps/ACCUM_ITER))

input1 = torch.arange(NUM_RANDOM_STEPS, 2*NUM_RANDOM_STEPS, 0.5, requires_grad=True)
input2 = torch.arange(NUM_RANDOM_STEPS, 2*NUM_RANDOM_STEPS, 0.5, requires_grad=True)
target = torch.Tensor([1]*2*NUM_RANDOM_STEPS)

loss_list = []

wandb.define_metric("custom_step")
wandb.define_metric("Loss", step_metric='custom_step')
wandb.define_metric("LearningRate", step_metric='custom_step')

for epoch_i in tqdm(range(EPOCHS)):
    total_train_loss = 0
    model.train()
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    model.zero_grad()

    for step, batch in enumerate(epoch_iterator):
        all_scores = model.forward(**batch)

        sign_tensor = torch.as_tensor([1]*all_scores[0].shape[0]).to(device)

        loss = dummy_criterion(input1.to(device), input2.to(device), target.to(device))

        for i in range(NUM_RANDOM_STEPS):
            if USE_CONTRASTIVE:
                con_loss = criterion(all_scores[i], all_scores[NUM_RANDOM_STEPS + i], sign_tensor)
                loss = loss + con_loss
                for j in range(i+1, NUM_RANDOM_STEPS):
                    tmp_loss = criterion(all_scores[i], all_scores[j], sign_tensor)
                    loss = loss + tmp_loss

        loss = loss / ACCUM_ITER

        if TEST:
            print('Loss')
            print(loss)

        if (epoch_i * len_dataloader+step+1)%LOG_EVERY==0:
            wandb_dict={
                'Loss': loss.mean().item(),
                'custom_step': step+1,
                'LearningRate': scheduler.get_lr()[0]
            }
            wandb.log(wandb_dict)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if ((step + 1) % ACCUM_ITER == 0) or (step + 1 == len(train_dataloader)):
            optimizer.step()
            model.zero_grad()
            scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    loss_list.append(avg_train_loss)
    print('Loss after epoch {} = {}'.format(epoch_i + 1, avg_train_loss))

    for child_model in model.children():
        child_model.model_layer.save_pretrained(save_model_folder + '_EPOCHS_' + str(epoch_i + 1))

    tokenizer.save_pretrained(save_model_folder + '_EPOCHS_' + str(epoch_i + 1))
    torch.save(model.state_dict(), os.path.join(save_model_folder, 'model_epochs_{}.pt'.format(epoch_i + 1)))

wandb.finish()
print(loss_list)

os.system('rm -r {}'.format('model_cache'))
