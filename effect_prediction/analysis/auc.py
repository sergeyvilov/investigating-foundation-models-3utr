#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import os

from sklearn.metrics import roc_auc_score

import scipy.stats


# In[2]:


data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/MLM/'


# In[16]:



# In[17]:

utr_tsv = data_dir + 'perbase_pred/model_scores_snp.tsv'
split = 'clinvar' # clinvar, gnomAD or eQTL

utr_tsv = sys.argv[1]
split = sys.argv[2]
output_name = sys.argv[3]

utr_variants = pd.read_csv(utr_tsv, sep='\t')

utr_variants.groupby('split').label.value_counts()


# In[18]:


utr_variants = utr_variants[utr_variants.split==split]


# In[19]:


models = ('Species-aware','Species-agnostic','DNABERT','NT-MS-v2-500M','13-mer','PhyloP100','PhyloP240')


# In[30]:


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

    print('Pref AUC')
    
    ref_auc = bootstrap_auc(utr_variants[f'{model_name}_ref'])
    
    if not 'PhyloP' in model_name:

        print('P alt inv AUC')
        inv_alt_auc = bootstrap_auc(np.log(1/utr_variants[f'{model_name}_alt']))
        print('P ratio AUC')
        ratio_auc = bootstrap_auc(np.log(utr_variants[f'{model_name}_ref']/utr_variants[f'{model_name}_alt']))

    else:
        
        inv_alt_auc = None
        ratio_auc = None

    roc_df.append((model_name, ref_auc, inv_alt_auc, ratio_auc))


# In[31]:


roc_df = pd.DataFrame(roc_df,columns=['model','ref_auc','inv_alt_auc','ratio_auc'])


# In[22]:


roc_df.to_csv(output_name, sep='\t',index=False)

