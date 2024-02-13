#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

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
from multiprocessing import Pool
import scipy


# In[2]:


data_dir = '/s/project/mll/sergey/effect_prediction/MLM/slutskin_2019/'


# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

input_params = dotdict({})

parser.add_argument("--model", help = 'embedding name, can be "MLM" "word2vec" "griesemer" or "Nmers" where N is an integer', type = str, required = True)

parser.add_argument("--output_dir", help = 'output folder', type = str, required = True)

parser.add_argument("--n_jobs", help = "number of CPU cores", default = 8, type = int, required = False)

parser.add_argument("--N_trials", help = "number of optuna trials", type = int, default = 100, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')
    
# In[4]:


mpra_df = pd.read_csv(data_dir + 'supl/Supplemental_Table_9.tab', sep='\t', skiprows=1, dtype={'Fold':str}, usecols=[0,1,2,3]) #sequence info

with open(data_dir + "embeddings_reversecompl/seq_len_5000/embeddings.pickle", 'rb') as f:
    mlm_embeddings = np.array(pickle.load(f))

#masked language model embeddings


supt2 = pd.read_csv(data_dir + 'supl/Supplemental_Table_2.tab', sep='\t', skiprows=1, dtype={'Fold':str})

# In[5]:


flt = (~mpra_df.Expression.isna()) & (mpra_df.ID.isin(supt2[supt2.Source=='K562'].ID))

flt = ~mpra_df.Expression.isna()

mpra_df = mpra_df[flt]
mlm_embeddings = mlm_embeddings[flt]

mpra_df = mpra_df.rename(columns={'Sequence':'seq'}).reset_index(drop=True)


# In[6]:


if input_params.model=='MLM':

    X = mlm_embeddings

elif 'mers' in input_params.model:
    
    k = int(input_params.model[0])
        
    kmerizer = Kmerizer(k=k)
    X = np.stack(mpra_df.seq.apply(lambda x: kmerizer.kmerize(x))) 
        
elif input_params.model=='word2vec':
        
    X = word2vec_model(mpra_df)

#X = np.hstack((X,np.expand_dims(mpra_df.min_free_energy.values,axis=1)))

y = mpra_df['Expression'].values


# In[7]:
def apply_SVR(args):
        
    fold_idx, test_hpp = args 

    test_idx = mpra_df[mpra_df.Fold==str(fold_idx)].index
    train_idx = mpra_df[(mpra_df.Fold!=str(fold_idx))&(mpra_df.Fold!='Test')].index

    pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                  sklearn.svm.SVR(**test_hpp))
    pipe.fit(X[train_idx],y[train_idx])

    R2_score = pipe.score(X[test_idx],y[test_idx])
        
    return R2_score


def hpp_search(X,y, mpra_df, cv_splits = 10):
    
    '''
    Perform Hyperparameter Search using OPTUNA Bayesian Optimisation strategy
    
    The bets hyperparameters should maximize coefficient of determination (R2)
    
    The hyperparameter range should first be adjused with grid search to make the BO algorithm converge in reasonable time
    '''


    def objective(trial):

        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-5, 1, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1, log=True)

        test_hpp = {'C':C, 'epsilon':epsilon, 'gamma':gamma}
        
        pool = Pool(processes=input_params.n_jobs,maxtasksperchild=5)

        cv_scores = []
        
        params = ((fold_idx, test_hpp) for fold_idx in range(cv_splits))
        
        for res in pool.imap(apply_SVR,params):
            cv_scores.append(res)
     
        pool.close()
        pool.join()
    
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.N_trials)
    
    best_params = study.best_params
    
    return best_params


# In[8]:




# In[9]:


def compute_metrics(y_true,y_pred):
    R2 = sklearn.metrics.r2_score(y_true,y_pred)
    Pearson_r = scipy.stats.pearsonr(y_true,y_pred)[0]
    return f'R2 {R2:.3f}, Pearson r: {Pearson_r:.3f}'


# In[14]:


best_hpp = hpp_search(X,y,mpra_df)

pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                    sklearn.svm.SVR(**best_hpp))

test_idx = mpra_df[mpra_df.Fold=='Test'].index
train_idx = mpra_df[(mpra_df.Fold!='Test')].index
        
X_train, X_test, y_train, y_test = X[train_idx,:],X[test_idx,:],y[train_idx],y[test_idx]

pipe.fit(X_train,y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

print(f'Train {compute_metrics(y_train, y_pred_train)}')
print(f'Test {compute_metrics(y_test, y_pred_test)}')


# In[11]:


os.makedirs(input_params.output_dir, exist_ok=True) #make output dir

with open(input_params.output_dir + '/best_model.pickle', 'wb') as f:
    pickle.dump(pipe, f)


# In[17]:


mpra_df.loc[train_idx,'y_pred'] = y_pred_train
mpra_df.loc[test_idx,'y_pred'] = y_pred_test


# In[19]:


mpra_df.to_csv(input_params.output_dir + '/all_predictions.tsv',sep='\t',index=None)




