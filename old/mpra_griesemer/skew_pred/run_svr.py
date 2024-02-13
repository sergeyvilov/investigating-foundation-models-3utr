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

parser.add_argument("--n_jobs", help = "number of CPU cores", default = 8, type = int, required = False)

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


# In[5]:

#Expression column to float
mpra_df['Expression'] = mpra_df[f'log2FoldChange_Skew_{input_params.cell_type}']

mpra_df['tag'] = mpra_df.apply(lambda x: 'ref' if x.oligo_id.endswith('_ref') else 'alt',axis=1)

assert all(mpra_df.loc[mpra_df.tag=='ref','mpra_variant_id'].values==
         mpra_df.loc[mpra_df.tag=='alt','mpra_variant_id'].values)
    
mpra_df.Expression = mpra_df.Expression.apply(lambda x:x.replace(',','.') if type(x)==str else x).astype(float)

enformer_dir = data_dir + 'enformer/predictions/'

data = {}

for pickle_file in os.listdir(enformer_dir):
    with open(enformer_dir+pickle_file,'rb') as f:
        data = data|pickle.load(f)
        
enformer_df = pd.DataFrame(data).T

idx = mpra_df.set_index(['mpra_variant_id','tag']).index

enformer_df = enformer_df.loc[idx].swaplevel() #get variants in exactly the same order as in mpra_df

enformer_log2fc = np.log2(enformer_df.loc['alt']/enformer_df.loc['ref']) #log2fc for all targets

enformer_log2fc = enformer_log2fc.fillna(enformer_log2fc.median())

assert all(enformer_log2fc.index==mpra_df[mpra_df.tag=='alt'].mpra_variant_id)

# In[7]:


def get_embeddings(mpra_df):

    if input_params.model=='MLM':

        X = mlm_embeddings[mpra_df.index]

    elif 'mers' in input_params.model:

        k = int(input_params.model[0])

        kmerizer = Kmerizer(k=k)
        X = np.stack(mpra_df.seq.apply(lambda x: kmerizer.kmerize(x))) 

    elif input_params.model=='word2vec':

        X = word2vec_model(mpra_df)

    elif input_params.model=='griesemer':

        X = minseq_model(mpra_df)

    X = np.hstack((X,np.expand_dims(mpra_df.min_free_energy.values,axis=1)))
    
    return X

def get_enformer_matrix(model):
 
    dnase_all_idx = np.array(np.arange(0,674)) 

    cage_all_idx = np.array(np.arange(4675,5313)) 

    chipseq_all_idx = np.array(np.arange(674,4675)) 

    if model == 'enformer_all_targets':
        
        X = enformer_log2fc.values
        
    elif model == 'enformer_summary':
        
        X = np.vstack((enformer_log2fc[dnase_all_idx].mean(axis=1),
                       enformer_log2fc[cage_all_idx].mean(axis=1),
                         enformer_log2fc[cage_all_idx].mean(axis=1))).T
        
    X = np.hstack((X,np.expand_dims(mpra_df[mpra_df.tag=='alt'].min_free_energy.values,axis=1),
    np.expand_dims(mpra_df[mpra_df.tag=='ref'].min_free_energy.values,axis=1)))
    
    return X

if not 'enformer' in input_params.model:
    
    X_ref = get_embeddings(mpra_df[mpra_df.tag=='ref'])
    X_alt = get_embeddings(mpra_df[mpra_df.tag=='alt'])

    X = np.hstack((X_ref,X_alt))
    
else:
    
    X = get_enformer_matrix(input_params.model)
    
y = mpra_df.loc[mpra_df.tag=='alt', 'Expression'].values
groups = mpra_df.loc[mpra_df.tag=='alt', 'group'].values


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


gss = sklearn.model_selection.GroupShuffleSplit(n_splits=input_params.N_splits, train_size=.9, random_state = input_params.seed) 

train_idx, test_idx = next(iter(gss.split(X, y, groups)))

X_train, X_test, y_train, y_test = X[train_idx,:],X[test_idx,:],y[train_idx],y[test_idx] #first split

#best_hpp = hpp_search(X_train,y_train,groups[train_idx],cv_splits = input_params.N_CVsplits) #get optimal hyperparameters
best_hpp = {'C': 0.03943153578419499, 'epsilon': 0.0712140417882623, 'gamma': 0.000232694021502066}

def apply_SVR(args):
    
    train_idx, test_idx = args

    pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                              sklearn.svm.SVR(**best_hpp))
    pipe.fit(X[train_idx,:],y[train_idx])  
    
    y_pred = np.full_like(y,np.NaN)
    
    y_pred[test_idx] = pipe.predict(X[test_idx,:])  
    
    r2 = sklearn.metrics.r2_score(y[test_idx], y_pred[test_idx])
    
    print('done')

    return y_pred, r2
 
def run_pool():
    
    all_res = []
    
    pool = Pool(processes=input_params.n_jobs,maxtasksperchild=5)

    #for train_idx, test_idx in gss.split(X,y,groups):
    #    pool.apply_async(apply_SVR,args =  (train_idx,test_idx), callback=log_result)
    
    for res in pool.imap(apply_SVR,gss.split(X,y,groups)):
        all_res.append(res)
     
    pool.close()
    pool.join()
    
    return all_res

print('running parallel')

all_res = run_pool()

preds, scores = zip(*all_res)

cv_res = np.vstack(preds)

cv_scores = pd.DataFrame({'round':range(input_params.N_splits),'scores':scores}|best_hpp)


# In[ ]:


os.makedirs(input_params.output_dir, exist_ok=True) #make output dir

cv_scores.to_csv(input_params.output_dir + '/cv_scores.tsv', sep='\t', index=None) #save scores

with open(input_params.output_dir + '/cv_res.npy', 'wb') as f:
    np.save(f, cv_res) #save predictions at each round
    np.save(f, y)


# In[ ]:




