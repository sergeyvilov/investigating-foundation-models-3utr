#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from collections import defaultdict

import os
import sys
import argparse

import pickle
import json

import re

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn

import sklearn.pipeline
import sklearn.model_selection
import sklearn.metrics

from sklearn.preprocessing import StandardScaler

import joblib
from joblib import parallel_backend

import optuna

sys.path.append("/home/icb/sergey.vilov/workspace/MLM/utils")
from mlp import *
from misc import dotdict

data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/half_life/agarwal_2022/'

parser = argparse.ArgumentParser("main.py")

input_params = dotdict({})

parser.add_argument("--model", help = 'model name, e.g. "effective_length" or "Nmers" where N is an integer', type = str, required = True)

parser.add_argument("--embeddings", help = 'embeddings file when a language model is used', type = str, default=None, required = False)

parser.add_argument("--output_name", help = 'output name', type = str, required = True)

parser.add_argument("--config", help = 'json file with hyperparameters', type = str, default=None, required = False)

parser.add_argument("--n_hpp_trials", help = "number of hpp search trials", type = int, default = 100, required = False)

parser.add_argument("--regressor", help = 'Ridge,SVR, or MLP', type = str, required = True)


#parser.add_argument("--n_folds", help = "number of CV splits in the outer loop", type = int, default = 10, required = False)

parser.add_argument("--cv_splits_hpp", help = "number of CV splits for hyperparameter search", type = int, default = 5, required = False)

parser.add_argument("--n_jobs", help = "number of simultaneous processes", type = int, default = 8, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')

print(f'Total CPUs available: {joblib.cpu_count()}')

folds_df = pd.read_csv(data_dir + 'agarwal_data/saluki_paper/Fig3_S4/binnedgenes.txt', sep='\t', usecols=[0,1],
                      names=['Fold','gene_id'], skiprows=1).set_index('gene_id') #folds as they are in Agarwal article

folds_df = folds_df-1 #to 0-based

features_df = pd.read_parquet(data_dir + 'preprocessing/seqFeatWithKmerFreqs_no5UTR.parquet.gz').set_index('GENE')

target_df = features_df[['HALFLIFE']]
features_df = features_df.drop(columns='HALFLIFE')

transcript_to_gene = pd.read_csv(data_dir + '../../UTR_coords/GRCh38_EnsembleCanonical_HGNC.tsv.gz', sep='\t', 
                                     names=['gene_id','transcript_id'], skiprows=1,usecols=[0,1]).set_index('transcript_id')

human_fasta = data_dir + '../../fasta/Homo_sapiens_rna.fa'

utr_df = defaultdict(str)

with open(human_fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            transcript_id = line[1:].split('.')[0]
        else:
            utr_df[transcript_id] += line.rstrip().upper()

utr_df = pd.DataFrame(utr_df.values(),
             index=transcript_to_gene.loc[utr_df.keys()].gene_id, 
             columns=['seq'])

data_df = [folds_df,target_df]

if input_params.model.startswith('BC3MS'):
    
    print('adding basic features')
    data_df.append(features_df.iloc[:,:8])

    print('adding codons')
    data_df.append(features_df[[x for x in features_df.columns if x.startswith('Codon.')]])

    print('adding SeqWeaver RBP binding (780)')
    for region in ('3pUTR','5pUTR','ORF'):
        df = pd.read_csv(data_dir + f'agarwal_data/human/SeqWeaver_predictions/{region}_avg.txt.gz', sep='\t').set_index('Group.1')
        data_df.append(df)

    print('miRNA target repression (319)')
    df = pd.read_csv(data_dir + 'agarwal_data/human/CWCS.txt.gz', sep='\t').set_index('GeneID')
    data_df.append(df)

if input_params.model=='3K' or input_params.model=='BC3MS':

    print("adding k-mer embeddings for 3'UTRs")
    data_df.append(features_df[[x for x in features_df.columns if x.startswith('3UTR.')]])

data_df = pd.concat(data_df,axis=1) #concat all features, except embeddings

data_df = data_df[~data_df.HALFLIFE.isna()]
data_df.fillna(0, inplace=True)

if input_params.embeddings!=None:
    print('adding language model embeddings')
    with open(input_params.embeddings,'rb') as f:
        X = pickle.load(f)
        print(f'number of sequences after filtering: {len(data_df)}')
        embeddings = np.vstack(X['embeddings'])
        print(f"embeddings size: {len(embeddings)}")
        embeddings_genes=transcript_to_gene.loc[[x.split('.')[0] for x in X['seq_names']]].gene_id
        data_df = data_df[data_df.index.isin(embeddings_genes)]
        embeddings_df = pd.DataFrame(embeddings, index=embeddings_genes,
                                    columns=[f'emb_{x}' for x in range(embeddings.shape[1])])
        data_df = data_df.join(embeddings_df)
        print(f'number of sequences overlapping with embeddings: {len(data_df)}')

X = data_df.drop(columns=['Fold','HALFLIFE']).values#all columns except HALFLIFE and fold

y = data_df['HALFLIFE'].values

folds = data_df['Fold'].astype(int).values

genes = data_df.index

del data_df

def hpp_search_mlp(X,y,forced_hpp={},cv_splits = 5):

    def objective(trial):

        p_dropout = trial.suggest_float("p_dropout", 0, 0.9, step=0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(1024,128,32),(512,128,16),(256,64,32),(128,64,32),(128,32,16),(64,32,16)])

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
            MLPRegressor(p_dropout=p_dropout, weight_decay=weight_decay,
                          hidden_layer_sizes=hidden_layer_sizes, **forced_hpp))

        cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, 
                    cv = cv_splits, scoring = 'r2', n_jobs = 1)

        av_score = cv_score.mean()

        return av_score

    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.n_hpp_trials, gc_after_trial=True)

    best_params = study.best_params

    best_params.update(forced_hpp)

    print('Using hyperparameters: ',best_params)

    return best_params
    

def hpp_search_svr(X,y,cv_splits = 5):

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
            cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, 
                     cv = cv_splits, scoring = 'r2', n_jobs = -1)

        av_score = cv_score.mean()

        return av_score

    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.n_hpp_trials)

    best_params = study.best_params

    return best_params

X_train, y_train = X[folds!=0],y[folds!=0]

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
    hpp_dict = hpp_search_svr(X_train, y_train,cv_splits = input_params.cv_splits_hpp)
    print('Using hyperparameters: ',hpp_dict)
    save_params(hpp_dict)

elif input_params.regressor=='MLP':

    forced_hpp = {'N_epochs':300,'lr':1e-4,'batch_size':1024} #pre-determined hpp, to reduce search time
    hpp_dict = hpp_search_mlp(X_train, y_train,forced_hpp=forced_hpp,cv_splits = input_params.cv_splits_hpp) #get optimal hyperparameters
    save_params(hpp_dict)
    
res_df = []

N_folds = folds.max()+1

print(f'Total folds: {N_folds}')

for fold in range(N_folds):
    
        print(f'Fold {fold}')
        
        X_train, X_test, y_train, y_test = X[folds!=fold],X[folds==fold],y[folds!=fold],y[folds==fold]

        if input_params.regressor == 'Ridge':
            pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), 
                                              sklearn.linear_model.RidgeCV(cv=input_params.cv_splits_hpp, alphas=10.**np.arange(-5,6)))

        elif input_params.regressor == 'Lasso':
            pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), 
                                              sklearn.linear_model.LassoCV(cv=input_params.cv_splits_hpp, n_alphas=50))
            
        elif input_params.regressor == 'SVR':
            pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                  sklearn.svm.SVR(**hpp_dict))

        elif input_params.regressor == 'MLP':
            pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                  MLPRegressor(**hpp_dict))
        
        pipe.fit(X_train,y_train)
                    
        y_pred = pipe.predict(X_test) 
                
        fold_res = np.vstack([np.ones((len(y_test),))*fold,genes[folds==fold],y_test,y_pred]).T

        res_df.append(fold_res)

res_df = np.vstack(res_df)
res_df = pd.DataFrame(res_df,columns=['fold','gene','y_true','y_pred'])

print('All Done')

res_df.to_csv(input_params.output_name, sep='\t', index=None)
