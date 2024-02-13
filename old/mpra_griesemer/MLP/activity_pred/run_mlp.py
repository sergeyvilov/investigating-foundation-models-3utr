#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
import sys

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn

import sklearn.pipeline 
import sklearn.model_selection
import sklearn.metrics

from sklearn.preprocessing import StandardScaler

import optuna

sys.path.append("/data/ouga/home/ag_gagneur/l_vilov/workspace/species-aware-DNA-LM/mpra_griesemer/utils") 

from models import *
from misc import dotdict


# In[2]:


data_dir = '/s/project/mll/sergey/effect_prediction/MLM/griesemer/'


# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

input_params = dotdict({})

parser.add_argument("--cell_type", help = "HMEC,HEK293FT,HEPG2,K562,GM12878,SKNSH", type = str, required = True)

parser.add_argument("--model", help = 'embedding name, can be "MLM" "word2vec" "griesemer" or "Nmers" where N is an integer', type = str, required = True)

parser.add_argument("--output_dir", help = 'output folder', type = str, required = True)

parser.add_argument("--N_trials", help = "number of optuna trials", type = int, default = 100, required = False)

parser.add_argument("--keep_first", help = "perform hpp search only at the first split, then use these hyperparameters", action='store_true', default = False, required = False)

parser.add_argument("--N_splits", help = "number of GroupShuffleSplits", type = int, default = 200, required = False)

parser.add_argument("--N_CVsplits", help = "number of CV splits for hyperparameter search", type = int, default = 5, required = False)

parser.add_argument("--seed", help = "seed fot GroupShuffleSplit", type = int, default = 1, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')
    
# In[4]:


mpra_df = pd.read_csv(data_dir + 'mpra_df.tsv', sep='\t') #sequence info

mlm_embeddings = np.load(data_dir + "embeddings/seq_len_5000/embeddings.npy") #masked language model embeddings

#Data Cleaning
# Take only SNP mutations
# Remove nan values in Expression column

is_snp = mpra_df.ref_allele.str.len() == mpra_df.alt_allele.str.len()

flt = mpra_df[f'log2FoldChange_Skew_{input_params.cell_type}'].isna()  | (~is_snp) | (mpra_df.stop_codon_dist>5000) #| mpra_df.oligo_id.str.contains('_ref$')

mpra_df = mpra_df[~flt]
mlm_embeddings = mlm_embeddings[~flt]


# In[5]:


#Expression column to float

mpra_df['Expression'] = mpra_df.apply(lambda x: x[f'log2FoldChange_Alt_{input_params.cell_type}'] if x.oligo_id.endswith('_alt') else x[f'log2FoldChange_Ref_{input_params.cell_type}'], axis=1)   
mpra_df.Expression = mpra_df.Expression.apply(lambda x:x.replace(',','.') if type(x)==str else x).astype(float)


# In[7]:


if input_params.model=='MLM':

    X = mlm_embeddings

elif 'mers' in input_params.model:
    
    k = int(input_params.model[0])
        
    kmerizer = Kmerizer(k=k)
    X = np.stack(mpra_df.seq.apply(lambda x: kmerizer.kmerize(x))) 
        
elif input_params.model=='word2vec':
        
    X = word2vec_model(mpra_df)

elif input_params.model=='griesemer':
        
    X = minseq_model(mpra_df)

X = np.hstack((X,np.expand_dims(mpra_df.min_free_energy.values,axis=1)))

y = mpra_df['Expression'].values
groups = mpra_df['group'].values


# In[8]:


def hpp_search(X,y,groups,cv_splits = 5):
    
    '''
    Perform Hyperparameter Search using OPTUNA Bayesian Optimisation strategy
    
    The bets hyperparameters should maximize coefficient of determination (R2)
    
    The hyperparameter range should first be adjused with grid search to make the BO algorithm converge in reasonable time
    '''

    def objective(trial):

        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-5, 1, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1, log=True)

        clf = sklearn.svm.SVR(C=C, epsilon=epsilon, gamma=gamma)

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),clf)

        cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, groups=groups, 
                     cv = sklearn.model_selection.GroupKFold(n_splits = cv_splits), scoring = 'r2', n_jobs = -1)
        
        av_score = cv_score.mean()
        
        return av_score
    
    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.N_trials)
    
    best_params = study.best_params
    
    return best_params


# In[ ]:


cv_res = np.zeros((input_params.N_splits,len(y))) #predictions for each point in each split
cv_res[:] = np.NaN 

cv_scores = [] #scores and best hyperparameters for each split

gss = sklearn.model_selection.GroupShuffleSplit(n_splits=input_params.N_splits, train_size=.9, random_state = input_params.seed) 

for round_idx, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        
        X_train, X_test, y_train, y_test = X[train_idx,:],X[test_idx,:],y[train_idx],y[test_idx]
        
        if round_idx==0 or input_params.keep_first==False:
            #perform only ones if input_params.keep_first==True
            best_hpp = hpp_search(X_train,y_train,groups[train_idx],cv_splits = input_params.N_CVsplits)
        
        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                              sklearn.svm.SVR(**best_hpp))
        
        pipe.fit(X_train,y_train)

        y_pred = pipe.predict(X_test)
                    
        cv_res[round_idx,test_idx] = y_pred
        
        cv_scores.append({'round':round_idx,'r2':sklearn.metrics.r2_score(y_test,y_pred)}|best_hpp)
        
cv_scores = pd.DataFrame(cv_scores)


# In[ ]:


os.makedirs(input_params.output_dir, exist_ok=True) #make output dir

cv_scores.to_csv(input_params.output_dir + '/cv_scores.tsv', sep='\t', index=None) #save scores

with open(input_params.output_dir + '/cv_res.npy', 'wb') as f:
    np.save(f, cv_res) #save predictions at each round
    np.save(f, y)


# In[ ]:




