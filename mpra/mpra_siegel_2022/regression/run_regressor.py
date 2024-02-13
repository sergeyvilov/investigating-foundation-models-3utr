#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import re

import os
import json

import sys

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn

import sklearn.pipeline
import sklearn.model_selection
import sklearn.metrics

from sklearn.preprocessing import StandardScaler

import joblib
from joblib import parallel_backend

from multiprocessing import Pool

import optuna

sys.path.append("/home/icb/sergey.vilov/workspace/MLM/mpra/utils")

from mlp import *
from models import *
from misc import dotdict

import scipy.stats

# In[2]:


data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/siegel_2022/'


# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

input_params = dotdict({})

parser.add_argument("--mpra_tsv", help = 'preprocessed MPRA dataframe', type = str, required = True)

parser.add_argument("--response", help = "steady_state or stability", type = str, required = True)

parser.add_argument("--model", help = 'model name, e.g. "effective_length" or "Nmers" where N is an integer', type = str, required = True)

parser.add_argument("--embeddings", help = 'embeddings file when a language model is used', type = str, default=None, required = False)

parser.add_argument("--regressor", help = 'SVR or Ridge', type = str, required = True)

parser.add_argument("--output_name", help = 'output name', type = str, required = True)

parser.add_argument("--config", help = 'json file with hyperparameters', type = str, default=None, required = False)

parser.add_argument("--onlyknown", help = "1 to use only SNP variants with dbSNP id", type = int, default = None, required = None)

parser.add_argument("--onlyref", help = "1 to use only reference sequences", type = int, default = None, required = None)

parser.add_argument("--onlyARE", help = "1 to use only ARE variants", type = int, default = 1, required = None)

parser.add_argument("--n_hpp_trials", help = "number of hpp search trials", type = int, default = 100, required = False)
#parser.add_argument("--keep_first", help = "perform hpp search only at the first split, then use these hyperparameters", action='store_true', default = False, required = False)

#parser.add_argument("--n_folds", help = "number of CV splits in the outer loop", type = int, default = 10, required = False)

parser.add_argument("--cv_splits_hpp", help = "number of CV splits for hyperparameter search", type = int, default = 5, required = False)

parser.add_argument("--seed", help = "seed fot GroupShuffleSplit", type = int, default = 1, required = False)

parser.add_argument("--n_jobs", help = "number of simultaneous processes", type = int, default = 8, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')

print(f'Total CPUs available: {joblib.cpu_count()}')


mpra_df = pd.read_csv(input_params.mpra_tsv, sep='\t') #sequence info

#Data Cleaning
# Take only SNP mutations
# Remove nan values in Expression column

if input_params.response == 'steady_state':
    mpra_df['Expression'] = mpra_df.ratios_T0_GC_resid
elif input_params.response == 'stability':
    mpra_df['Expression'] = mpra_df.ratios_T4T0_GC_resid

#flt = (mpra_df.Expression.isna()) | (mpra_df.ARE_length_perfect.isna()) | (mpra_df.stop_codon_dist.isna()) | (mpra_df.stop_codon_dist>5000) | (~mpra_df.issnp.astype(bool))

flt = (mpra_df.Expression.isna()) | (mpra_df.stop_codon_dist.isna()) | (~mpra_df.issnp.astype(bool))

if input_params.onlyknown:
    flt = (flt)|(mpra_df.SNP.isna())

if input_params.onlyARE:
    flt = (flt)|(mpra_df.ARE_length_perfect.isna())

if input_params.onlyref:
    flt = (flt)|(mpra_df.iscontrol==1)

mpra_df = mpra_df[~flt]

if input_params.embeddings!=None:
    with open(input_params.embeddings,'rb') as f:

        X = pickle.load(f)
        print(f'number of sequences after filtering: {len(mpra_df)}')
        print(f"embeddings size: {len(X['embeddings'])}")
        seq_indices=[int(seq_name.replace('id_','')) for seq_name in X['seq_names']]
        mpra_df = mpra_df[mpra_df.index.isin(seq_indices)]
        embeddings = {seq_name:emb for seq_name,emb in zip(X['seq_names'], X['embeddings'])}
        X = [embeddings[f'id_{seq_idx}'] for seq_idx in mpra_df.index]
        X = np.array(X)
        print(f'number of sequences overlapping with embeddings: {len(X)}')
        #X = X[mpra_df.index]

elif 'mers' in input_params.model:

    k = int(input_params.model[0])

    kmerizer = Kmerizer(k=k)
    X = np.stack(mpra_df.seq.apply(lambda x: kmerizer.kmerize(x)))

elif input_params.model=='word2vec':

    X = word2vec_model(mpra_df)

elif input_params.model=='effective_length':

    X = mpra_df.ARE_registration_perfect + mpra_df.ARE_length_perfect
    X = np.expand_dims(X.values,1)

mpra_df['group'] = mpra_df.region.apply(lambda x:x.split('|')[1].split(':')[0])

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

        C = trial.suggest_float("C", 1e-2, 1, log=True)
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

        group_kfold = sklearn.model_selection.GroupKFold(n_folds=input_params.cv_splits_hpp).split(X[train_idx],y[train_idx],groups[train_idx])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.LassoCV(cv=group_kfold, alphas=10.**np.arange(-6,0)))


    elif input_params.regressor=='Ridge':

        #predict with Ridge
        #use inner CV loop to adjust alpha

        group_kfold = sklearn.model_selection.GroupKFold(n_folds=input_params.cv_splits_hpp).split(X[train_idx],y[train_idx],groups[train_idx])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.RidgeCV(cv=group_kfold, alphas=10.**np.arange(-5,6)))

    pipe.fit(X[train_idx],y[train_idx])

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
