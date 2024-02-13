#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re

import os
import sys
import pickle
import json
import joblib

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn
import optuna

import sklearn.pipeline
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegressionCV

from multiprocessing import Pool

sys.path.append("/home/icb/sergey.vilov/workspace/MLM/mpra/utils")

from mlp import *
from misc import dotdict

import argparse

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--data_tsv", help = 'dataframe with variants info', type = str, required = True)

parser.add_argument("--embeddings", help = 'embeddings file from a language model', type = str, default=None, required = False)

parser.add_argument("--subset", help = "clivar,gnomad,eqtl-susie,eqtl-grasp", type = str, required = True)

parser.add_argument("--classifier", help = 'LogisticRegression or MLP', type = str, required = True)

parser.add_argument("--output_name", help = 'output name', type = str, required = True)

parser.add_argument("--n_hpp_trials", help = "number of hpp search trials", type = int, default = 100, required = False)

parser.add_argument("--config", help = 'json file with hyperparameters', type = str, default=None, required = False)

#parser.add_argument("--keep_first", help = "perform hpp search only at the first split, then use these hyperparameters", action='store_true', default = False, required = False)

parser.add_argument("--merge_embeddings", help = "merge ref and alt embeddings", type=int, default = 0, required = False)

parser.add_argument("--upsample_to_majority_class", help = "upsample to majority class in case of imbalance", action='store_true', default = False, required = False)

parser.add_argument("--n_folds", help = "number of CV splits in the outer loop", type = int, default = 10, required = False)

parser.add_argument("--cv_splits_hpp", help = "number of CV splits for hyperparameter search", type = int, default = 5, required = False)

parser.add_argument("--seed", help = "seed for shuffling train data", type = int, default = None, required = False)

parser.add_argument("--n_jobs", help = "number of processes for multiprocessing (MLP excluded)", type = int, default = 8, required = False)


input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')

# In[4]:

print(f'Total CPUs available: {joblib.cpu_count()}')

data_df = pd.read_csv(input_params.data_tsv, sep='\t')

data_df = data_df[data_df.split==input_params.subset]

with open(input_params.embeddings, 'rb') as f:
    data = pickle.load(f)
    seq_names, embeddings = data['seq_names'], data['embeddings']
    embeddings_df = []
    for idx in range(0,len(embeddings),2):
        assert seq_names[idx]==seq_names[idx+1].replace('alt','ref')
        emb_ref, emb_alt = embeddings[idx], embeddings[idx+1]
        varname = seq_names[idx].replace('_ref','').split('_')
        embeddings_df.append((varname[0],int(varname[1]),varname[2],varname[3],emb_ref,emb_alt))
    embeddings_df = pd.DataFrame(embeddings_df, columns=['chrom','pos','ref','alt','emb_ref','emb_alt'])

data_df = data_df.merge(embeddings_df, how='left')

data_df.reset_index(drop=True, inplace=True) #important for joining results at the end

if input_params.merge_embeddings:
    X = data_df[['emb_ref','emb_alt']].values
else:
    X = data_df.emb_alt.values

X = np.array(X.tolist()).reshape(len(X),-1) #contatenate ref and alt embeddings for each variant

y = data_df['label'].values

def hpp_search_mlp(X,y,forced_hpp={},cv_splits = 5):

    X, y = prepare_train_data(X,y)

    def objective(trial):
        
        p_dropout = trial.suggest_float("p_dropout", 0, 0.9, step=0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(1024,128,32),(512,128,16),(256,64,32),(128,64,32),(128,32,16),(64,32,16)])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
            MLPClassifier(p_dropout=p_dropout, weight_decay=weight_decay, 
                          hidden_layer_sizes=hidden_layer_sizes, **forced_hpp))
        
        cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, verbose=10,
                    cv = sklearn.model_selection.StratifiedKFold(n_splits = cv_splits), 
                                                           scoring = 'roc_auc', n_jobs = 1)
        
        av_score = cv_score.mean()
        
        return av_score

    #optuna.logging.set_verbosity(optuna.logging.DEBUG)
    
    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.n_hpp_trials, gc_after_trial=True)
    
    best_params = study.best_params

    best_params.update(forced_hpp)

    print('Using hyperparameters: ',best_params)

    return best_params

#def hpp_search_mlp(X,y,forced_hpp={},cv_splits = 5):
#
#    X, y = prepare_train_data(X,y)
#
#    param_distributions = {
#            'MLP__p_dropout':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#            'MLP__weight_decay':[0,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2],
#            'MLP__hidden_layer_sizes':[(1024,128,32),(512,128,16),(256,64,32),(128,64,32),(128,32,16),(64,32,16)],
#        }
#
#    pipe = sklearn.pipeline.Pipeline(steps=[
#         ('StandardScaler', sklearn.preprocessing.StandardScaler()),
#         ('MLP', MLPClassifier(**forced_hpp))
#        ])
#
#    clf = sklearn.model_selection.RandomizedSearchCV(pipe, param_distributions, scoring = 'roc_auc',
#                    cv = sklearn.model_selection.StratifiedKFold(n_splits = cv_splits),  n_jobs = 1, n_iter=input_params.n_hpp_trials, random_state=1, verbose=10)
#
#    search = clf.fit(X, y)
#
#    print(f'Best CV score: {search.best_score_}')
#
#    best_params = search.best_params_
#
#    best_params = {re.sub('.*__','',k):v for k,v in best_params.items()}
#
#    best_params.update(forced_hpp)
#
#    print('Using hyperparameters: ',best_params)
#
#    return best_params

def upsample_to_majority_class(X,y):

    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_neg>n_pos:
        new_idx  = np.random.choice(pos_idx, size=n_neg-n_pos)
    else:
        new_idx  = np.random.choice(neg_idx, size=n_pos-n_neg)

    X_resampled = np.vstack((X,X[new_idx]))
    y_resampled = np.hstack((y,y[new_idx]))

    return X_resampled, y_resampled

if input_params.seed is not None:
    np.random.seed(input_params.seed)

def prepare_train_data(X,y):

    if input_params.upsample_to_majority_class:
        X, y = upsample_to_majority_class(X,y)

    shuffled_idx = np.random.permutation(len(X))
    X, y = X[shuffled_idx], y[shuffled_idx]

    return X,y

skf = sklearn.model_selection.StratifiedKFold(n_splits=input_params.n_folds)

train_idx, _ = next(iter(skf.split(X, y)))

def save_params(hpp_dict):
    params_path = os.path.splitext(input_params.output_name)[0] + '.config.json'
    print(f'Saving hyperparameters to {params_path}')
    with open(params_path, 'w') as f:
          json.dump(hpp_dict, f, ensure_ascii=False)

if input_params.config:
    print(f'Loading hyperparameters from: {input_params.config}')
    with open(input_params.config, 'r') as f:
        hpp_dict = json.load(f)
    print('Using hyperparameters: ',hpp_dict)

elif input_params.classifier=='MLP':

    forced_hpp = {'N_epochs':300,'lr':1e-4,'batch_size':1024} #pre-determined hpp, to reduce search time
    hpp_dict = hpp_search_mlp(X[train_idx],y[train_idx],forced_hpp=forced_hpp,cv_splits = input_params.cv_splits_hpp) #get optimal hyperparameters
    save_params(hpp_dict)


def apply_regression(args):

    fold, (train_idx, test_idx) = args

    print(f'predicting with {input_params.classifier}')

    if input_params.classifier=='LogisticRegression':

        kfold = sklearn.model_selection.StratifiedKFold(n_splits=input_params.cv_splits_hpp).split(X[train_idx],y[train_idx])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),LogisticRegressionCV(
                cv = kfold, Cs=10.**np.arange(-5,6)))

    X_train, y_train = prepare_train_data(X[train_idx],y[train_idx])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict_proba(X[test_idx])[:,1]

    print('done')

    fold_vector = [fold]*len(test_idx)

    return list(zip(fold_vector, test_idx, y_pred))


total_splits = skf.get_n_splits(X,y)

def run_pool():

    all_res = []

    pool = Pool(processes=input_params.n_jobs,maxtasksperchild=5)

    for fold,res in enumerate(pool.imap(apply_regression,zip(range(total_splits),skf.split(X,y)) )):
        all_res.extend(res)
        print(f'{fold+1}/{total_splits} folds processed')

    pool.close()
    pool.join()

    return all_res

if input_params.classifier!='MLP':

    print('running parallel')

    all_res = run_pool()

else:

    all_res = []

    pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                          MLPClassifier(**hpp_dict))


    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        X_train, y_train = prepare_train_data(X[train_idx],y[train_idx])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X[test_idx])[:,1]

        fold_vector = [fold]*len(test_idx)

        all_res.extend(list(zip(fold_vector, test_idx, y_pred)))

        print(f'{fold+1}/{total_splits} folds processed')


all_folds, all_idx, all_preds = zip(*all_res)
all_res = pd.DataFrame({'y_pred':all_preds,'fold':all_folds},index=all_idx)

data_df = data_df.join(all_res)

print('All Done')

data_df[['chrom','pos','ref','alt','split','label','y_pred']].to_csv(input_params.output_name, sep='\t', index=None)
