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
from joblib import parallel_backend

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn

import sklearn.pipeline
import sklearn.model_selection
import sklearn.metrics

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

import optuna

sys.path.append("/home/icb/sergey.vilov/workspace/MLM/utils")

from mlp import *
from models import *
from misc import dotdict


# In[2]:



# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

input_params = dotdict({})

parser.add_argument("--mpra_tsv", help = 'preprocessed MPRA dataframe', type = str, required = True)

parser.add_argument("--embeddings", help = 'embeddings file when a language model is used', type = str, default=None, required = False)

parser.add_argument("--cell_type", help = "HMEC,HEK293FT,HEPG2,K562,GM12878,SKNSH", type = str, required = True)

parser.add_argument("--model", help = 'embedding name, e.g. "dnabert" "ntrans-v2-250m" "griesemer" or "Nmers" where N is an integer', type = str, required = True)

parser.add_argument("--config", help = 'json file with hyperparameters', type = str, default=None, required = False)

parser.add_argument("--regressor", help = 'Ridge,SVR, or MLP', type = str, required = True)

parser.add_argument("--output_name", help = 'output name', type = str, required = True)

parser.add_argument("--n_hpp_trials", help = "number of hpp search trials", type = int, default = 1000, required = False)
#parser.add_argument("--keep_first", help = "perform hpp search only at the first split, then use these hyperparameters", action='store_true', default = False, required = False)

#parser.add_argument("--n_folds", help = "number of CV splits in the outer loop", type = int, default = 10, required = False)

parser.add_argument("--cv_splits_hpp", help = "number of CV splits for hyperparameter search", type = int, default = 5, required = False)

parser.add_argument("--seed", help = "seed fot GroupShuffleSplit", type = int, default = 1, required = False)

parser.add_argument("--n_jobs", help = "number of simultaneous processes", type = int, default = 8, required = False)


input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')

# In[4]:

print(f'Total CPUs available: {joblib.cpu_count()}')

mpra_df = pd.read_csv(input_params.mpra_tsv, sep='\t') #sequence info

#Data Cleaning
# Take only SNP mutations
# Remove nan values in Expression column

is_snp = mpra_df.ref.str.len() == mpra_df.alt.str.len()

flt = mpra_df[f'log2FoldChange_Ref_{input_params.cell_type}'].isna() | mpra_df[f'log2FoldChange_Alt_{input_params.cell_type}'].isna()  | (~is_snp) 

mpra_df = mpra_df[~flt]

#Expression column to float

mpra_df['Expression'] = mpra_df.apply(lambda x: x[f'log2FoldChange_Alt_{input_params.cell_type}'] if x.oligo_id.endswith('_alt') else x[f'log2FoldChange_Ref_{input_params.cell_type}'], axis=1)
mpra_df.Expression = mpra_df.Expression.apply(lambda x:x.replace(',','.') if type(x)==str else x).astype(float)

if input_params.embeddings!=None:
    with open(input_params.embeddings,'rb') as f:
        X = pickle.load(f)
        print(f'number of sequences after filtering: {len(mpra_df)}')
        print(f"embeddings size: {len(X['embeddings'])}")
        mpra_df = mpra_df[mpra_df.oligo_id.isin(X['seq_names'])]
        embeddings = {seq_name:emb for seq_name,emb in zip(X['seq_names'], X['embeddings'])}
        X = [embeddings[seq_name] for seq_name in mpra_df.oligo_id.values]
        X = np.array(X)
        print(f'number of sequences overlapping with embeddings: {len(X)}')

elif 'mers' in input_params.model:

    k = int(input_params.model[0])

    kmerizer = Kmerizer(k=k)
    X = np.stack(mpra_df.seq.apply(lambda x: kmerizer.kmerize(x)))

elif input_params.model=='word2vec':

    X = word2vec_model(mpra_df)

elif input_params.model=='griesemer':

    X = minseq_model(mpra_df)

#X = np.hstack((X,np.expand_dims(mpra_df.min_free_energy.values,axis=1)))

mpra_df.reset_index(drop=True, inplace=True) #important for joining results at the end

y = mpra_df['Expression'].values
groups = mpra_df['group'].values


def hpp_search_svr(X,y,groups,cv_splits = 5):

    '''
    Perform Hyperparameter Search using OPTUNA Bayesian Optimisation strategy

    The bets hyperparameters should maximize coefficient of determination (R2)

    The hyperparameter range should first be adjused with grid search to make the BO algorithm converge in reasonable time
    '''

    def objective(trial):

        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-5, 1, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1, log=True)

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                  sklearn.svm.SVR(C=C, epsilon=epsilon, gamma=gamma))

        with parallel_backend('multiprocessing', n_jobs=input_params.n_jobs):
            cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, groups=groups,
                     cv = sklearn.model_selection.GroupKFold(n_splits = cv_splits), scoring = 'r2', n_jobs = -1)

        av_score = cv_score.mean()

        return av_score

    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.n_hpp_trials)

    best_params = study.best_params

    return best_params

def hpp_search_mlp(X,y,groups,forced_hpp={},cv_splits = 5):

    def objective(trial):

        p_dropout = trial.suggest_float("p_dropout", 0, 0.9, step=0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(1024,128,32),(512,128,16),(256,64,32),(128,64,32),(128,32,16),(64,32,16)])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
            MLPRegressor(p_dropout=p_dropout, weight_decay=weight_decay,
                          hidden_layer_sizes=hidden_layer_sizes, **forced_hpp))

        cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, groups=groups,
                    cv = sklearn.model_selection.GroupKFold(n_splits = cv_splits),
                                                           scoring = 'r2', n_jobs = 1)

        av_score = cv_score.mean()

        return av_score

    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.n_hpp_trials, gc_after_trial=True)

    best_params = study.best_params

    best_params.update(forced_hpp)

    print('Using hyperparameters: ',best_params)

    return best_params

# def hpp_search_mlp(X,y,groups,forced_hpp={},cv_splits = 5):
#
#     #X, y = prepare_train_data(X,y)
#
#     param_distributions = {
#             'MLP__p_dropout':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#             'MLP__weight_decay':[0,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2],
#             'MLP__hidden_layer_sizes':[(1024,128,32),(512,128,16),(256,64,32),(128,64,32),(128,32,16),(64,32,16)],
#         }
#
#     pipe = sklearn.pipeline.Pipeline(steps=[
#          #('StandardScaler', sklearn.preprocessing.StandardScaler()),
#          ('MLP', MLPRegressor(**forced_hpp))
#         ])
#
#     clf = sklearn.model_selection.RandomizedSearchCV(pipe, param_distributions, scoring = 'r2',
#                     cv = sklearn.model_selection.GroupKFold(n_splits = cv_splits),  n_jobs = 1, n_iter=input_params.n_hpp_trials, random_state=1, verbose=10)
#
#     search = clf.fit(X, y, groups=groups)
#
#     print(f'Best CV score: {search.best_score_}')
#
#     best_params = search.best_params_
#
#     best_params = {re.sub('.*__','',k):v for k,v in best_params.items()}
#
#     best_params.update(forced_hpp)
#
#     print('Using hyperparameters: ',best_params)
#
#     return best_params


gss = sklearn.model_selection.LeaveOneGroupOut()

train_idx, _ = next(iter(gss.split(X, y, groups)))

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
    save_params(hpp_dict)

elif input_params.regressor=='SVR':
    hpp_dict = hpp_search_svr(X[train_idx],y[train_idx],groups[train_idx],cv_splits = input_params.cv_splits_hpp)
    print('Using hyperparameters: ',hpp_dict)
    save_params(hpp_dict)

elif input_params.regressor=='MLP':

    forced_hpp = {'N_epochs':300,'lr':1e-4,'batch_size':1024} #pre-determined hpp, to reduce search time
    hpp_dict = hpp_search_mlp(X[train_idx],y[train_idx],groups[train_idx],forced_hpp=forced_hpp,cv_splits = input_params.cv_splits_hpp) #get optimal hyperparameters
    save_params(hpp_dict)


def apply_regression(args):

    train_idx, test_idx = args

    print(f'predicting with {input_params.regressor}')

    if input_params.regressor=='SVR':

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                  sklearn.svm.SVR(**hpp_dict))

    elif input_params.regressor=='Lasso':

        #predict with Lasso
        #use inner CV loop to adjust alpha

        group_kfold = sklearn.model_selection.GroupKFold(n_splits=input_params.cv_splits_hpp).split(X[train_idx],y[train_idx],groups[train_idx])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.LassoCV(cv=group_kfold, alphas=10.**np.arange(-6,0)))


    elif input_params.regressor=='Ridge':

        #predict with Ridge
        #use inner CV loop to adjust alpha

        group_kfold = sklearn.model_selection.GroupKFold(n_splits=input_params.cv_splits_hpp).split(X[train_idx],y[train_idx],groups[train_idx])
            
        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.RidgeCV(cv=group_kfold, alphas=10.**np.arange(1,5)))

    pipe.fit(X[train_idx],y[train_idx])

    if input_params.regressor == 'Ridge':
            print(f'Best alpha: {pipe[1].alpha_}')

    y_pred = pipe.predict(X[test_idx])

    print('done')

    return list(zip(test_idx, y_pred))


total_splits = gss.get_n_splits(X,y,groups)

def run_pool():

    all_res = []

    pool = Pool(processes=input_params.n_jobs,maxtasksperchild=5)

    for fold,res in enumerate(pool.imap(apply_regression,gss.split(X,y,groups))):
        all_res.extend(res)
        print(f'{fold+1}/{total_splits} splits processed')

    pool.close()
    pool.join()

    return all_res


if input_params.regressor!='MLP':

    print('running parallel')

    all_res = run_pool()

else:

    all_res = []

    pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),MLPRegressor(**hpp_dict))

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):

        pipe.fit(X[train_idx],y[train_idx])

        y_pred = pipe.predict(X[test_idx])

        all_res.extend(list(zip(test_idx, y_pred)))

        print(f'{fold+1}/{total_splits} splits processed')


all_idx, all_preds = zip(*all_res)
all_res = pd.Series(all_preds,index=all_idx,name='y_pred')

print('All Done')

mpra_df.join(all_res).to_csv(input_params.output_name, sep='\t', index=None)
