import numpy as np
import pandas as pd
import sys
import os
import matplotlib

from sklearn.metrics import roc_auc_score

import scipy.stats


utr_tsv = sys.argv[1]
split = sys.argv[2]
output_name = sys.argv[3]

utr_variants = pd.read_csv(utr_tsv, sep='\t')

utr_variants = utr_variants[utr_variants.split==split]

models = ('StateSpace', 'StateSpace-SA','13-mer', 'DNABERT2', 'DNABERT2-3UTR','PhyloP-100way', 'PhyloP-241way',
          'DNABERT', 'DNABERT-3UTR', 'NTv2-250M','NTv2-250M-3UTR')

# In[30]:

scores = ['pref','palt_inv','pratio','l1','l2','dot','cosine', 
          'loss_alt','loss_diff','LogisticRegression','MLP']

from scipy.stats import bootstrap

def bootstrap_auc(score):

    y_true = utr_variants.label[~score.isna()].values
    y_pred = score[~score.isna()].values
    
    bs = bootstrap((y_true, y_pred),statistic=lambda x,y:roc_auc_score(x,y),
                   vectorized=False, paired=True,n_resamples=1000)

    auc = roc_auc_score(y_true,y_pred)
    auc_err = np.diff(bs.confidence_interval)/2
    return auc, auc_err[0]

roc_df = []

for model_name in models:

    print(model_name)

    model_scores = []
        
    for score_name in scores:
        if score_name == 'pref':
            if f'{model_name}-pref' not in utr_variants.columns:
                model_scores.append((None,None))
                continue
            score = utr_variants[f'{model_name}-pref']
        elif score_name == 'palt_inv':
            if f'{model_name}-palt' not in utr_variants.columns:
                model_scores.append((None,None))
                continue
            score = np.log(1/utr_variants[f'{model_name}-palt'])
        elif score_name == 'pratio':
            if f'{model_name}-palt' not in utr_variants.columns:
                model_scores.append((None,None))
                continue
            score = np.log(utr_variants[f'{model_name}-pref']/utr_variants[f'{model_name}-palt'])
        elif score_name == 'loss_diff':
            if f'{model_name}-loss_alt' not in utr_variants.columns:
                model_scores.append((None,None))
                continue
            score = utr_variants[f'{model_name}-loss_alt']-utr_variants[f'{model_name}-loss_ref']
        else:
            if f'{model_name}-{score_name}' not in utr_variants.columns:
                model_scores.append((None,None))
                continue
            score = utr_variants[f'{model_name}-{score_name}']
        print(score_name)
        bootstrap_est = bootstrap_auc(score)
        model_scores.append(bootstrap_est)
        
    roc_df.append((model_name, *model_scores))


roc_df = pd.DataFrame(roc_df,columns=['model'] + scores)

# In[22]:


roc_df.to_csv(output_name, sep='\t',index=False)

