from itertools import compress, permutations
from pathlib import Path
from random import shuffle
from scipy.spatial.distance import cdist
from sympy.combinatorics import Permutation
from tqdm import tqdm

import numpy as np
import os
import pickle
import random


def load_from_pickle(file_path, mode='rb'):
    with open(file_path, mode) as f:
        file = pickle.load(f)
    return file


def save_as_pickle(file_path, file, mode='wb'):
    with open(file_path, mode) as f:
        pickle.dump(file, f)


def get_permutation_list(hp):
    HAMMING_SELECTION = hp['HAMMING_SELECTION']
    PERMUTATION_HEURISTIC = hp['PERMUTATION_HEURISTIC'] 
    RECIPES_STEPS = hp['RECIPES_STEPS'] 
    TOTAL_NUM_PERMUTATIONS = hp['TOTAL_NUM_PERMUTATIONS'] 

    fname = f'permutation_{PERMUTATION_HEURISTIC}_{RECIPES_STEPS}'
    files = list(Path('.').iterdir())
    if len(files)==0:
        sfname = None
    elif len(files)==1:
        if  fname in str(files[0]):
            sfname = str(files[0])
    else:
        flist = [w for w in files if fname in str(w)]
        if len(flist)==1:
            sfname = str(flist[0])
        else:
            sfname = None

    permutation_list = []
    num_perm_to_process = TOTAL_NUM_PERMUTATIONS
    if sfname is not None:
        print(f"{sfname} exists") 
        permutation_list = load_from_pickle(sfname)
        num_processed_perm = int(sfname.split('_')[-1][:-len('.pickle')])
        num_perm_to_process -= num_processed_perm

    if num_perm_to_process<=0: 
        permutation_list = permutation_list[:TOTAL_NUM_PERMUTATIONS]
    else:
        if sfname is not None:
            print(f"deleteing {sfname}")
            os.remove(sfname)
        print(f'processing {num_perm_to_process} elements')
        if PERMUTATION_HEURISTIC=='random':
            permutation_set = set(permutation_list)
            range_list = list(range(RECIPES_STEPS))
            while len(permutation_set) < num_perm_to_process:
                shuffle(range_list)
                permutation_set.add(tuple(range_list)) 
            permutation_list_random = list(permutation_set)
            permutation_list = list(permutation_set)

        # https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/select_permutations.py
        if PERMUTATION_HEURISTIC=='hamming':  
            print(f"HAMMING_SELECTION: {HAMMING_SELECTION}")  
            P_hat = np.array(list(permutations(list(range(RECIPES_STEPS)), RECIPES_STEPS)))
            n = P_hat.shape[0]
            
            for i in tqdm(range(num_perm_to_process), desc='permutation set'):
                if i==0:
                    j = np.random.randint(n)
                    if len(permutation_list)==0:
                        P = np.array(P_hat[j]).reshape([1,-1])
                    else:
                        P = np.concatenate([permutation_list,P_hat[j].reshape([1,-1])],axis=0)
                else:
                    P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
                
                P_hat = np.delete(P_hat,j,axis=0)
                D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
                
                if HAMMING_SELECTION=='max':
                    j = D.argmax()
                else:
                    m = int(D.shape[0]/2)
                    S = D.argsort()
                    j = S[np.random.randint(m-10,m+10)]
            permutation_list_hamming = P
            permutation_list = P
            del(P_hat)
            del(P)

        fname += f"_{TOTAL_NUM_PERMUTATIONS}.pickle" 
        print(f"saving {fname}")
        save_as_pickle(f'{fname}', permutation_list)
    return permutation_list


def get_training_data(hp):
    RECIPES_FILE_PATH = hp['RECIPES_FILE_PATH']
    RECIPES_STEPS = hp['RECIPES_STEPS']
    NUM_DATA = hp['NUM_DATA']
    SEED = hp['SEED']
    TOTAL_NUM_PERMUTATIONS = hp['TOTAL_NUM_PERMUTATIONS']

    random.seed(SEED)
    np.random.seed(SEED)

    recipes = load_from_pickle(RECIPES_FILE_PATH)

    recipes_filtered = [w for w in recipes if len(w)==RECIPES_STEPS]

    print(f"number of recipes with {RECIPES_STEPS} steps: {len(recipes_filtered)}")
    del(recipes)

    if len(recipes_filtered)>NUM_DATA:
        recipes_filtered = random.sample(recipes_filtered, NUM_DATA)
    print(f"len(recipes_filtered): {len(recipes_filtered)}")

    permutation_list = get_permutation_list(hp)

    # permute recipes
    num_perm_per_recipe = int(NUM_DATA/len(recipes_filtered))
    print(f"num_perm_per_recipe: {num_perm_per_recipe}")

    temp_list = []
    for i in range(num_perm_per_recipe):
        temp_list += recipes_filtered

    recipes_cache = random.sample(recipes_filtered, NUM_DATA - len(temp_list))
    temp_list += recipes_cache

    idx_list = range(TOTAL_NUM_PERMUTATIONS)
    labels = []
    recipes_permuted = []
    for doc in tqdm(temp_list):
        idx = random.choice(idx_list)
        perm = permutation_list[idx]
        perm = Permutation(perm)
        labels.append(idx)
        recipes_permuted.append(perm(doc))

    recipes_documents = [' '.join(w) for w in recipes_permuted]
    print(f"\nlen(recipes_documents): {len(recipes_documents)}")

    return recipes_documents, labels, permutation_list
