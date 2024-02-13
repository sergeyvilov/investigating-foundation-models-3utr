#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import modiscolite

from collections import defaultdict


# In[2]:


data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/MLM/'


# In[3]:


human_fasta = data_dir + 'fasta/240_species/species/Homo_sapiens.fa' #3'UTR on hegative strand should already be reversed

human_utr = defaultdict(str)

with open(human_fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            seq_name = line[1:].split(':')[0]
        else:
            human_utr[seq_name] += line.rstrip().upper()


# In[4]:


model_path = 'motif_predictions/species_agnostic/probas'


# In[5]:


with open(data_dir + model_path + '.pickle','rb') as f:
   model_res = dict(pickle.load(f))


# In[10]:


all_probas = []
seq = ''

for seq_name, probas in model_res.items():
    seq += human_utr[seq_name]
    all_probas.append(probas)

all_probas = np.vstack(all_probas) #Lx4
all_probas = np.expand_dims(all_probas,0) # 1xLx4


# In[7]:


mapping = {'A':0,'C':1,'G':2,'T':3}

seq_num = [mapping[base] for base in seq]

nb_classes = 4
targets = np.array([seq_num]).reshape(-1)
seq_one_hot = np.eye(nb_classes)[targets].astype(np.float32) #Lx4
seq_one_hot = np.expand_dims(seq_one_hot,0) # 1xLx4


# In[11]:


#all_probas = all_probas*np.log((all_probas+1e-10)/all_probas.mean()) # CAUSES ERROR#

all_probas = all_probas/all_probas.sum(2, keepdims=True)
print(f'Subtracting median value {np.median(all_probas)}')
all_probas = all_probas-np.median(all_probas)


# In[13]:


#np.save('seq_one_hot.npy', seq_one_hot)
#np.save('probas_norm.npy', all_probas)


# In[ ]:


pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=all_probas,
    one_hot=seq_one_hot,
    max_seqlets_per_metacluster=2000,  # default is 2000
    sliding_window_size=5,
    flank_size=2,
    target_seqlet_fdr=0.05,
    n_leiden_runs=3,  # default is 2
    verbose=True,
)


# In[ ]:


modiscolite.io.save_hdf5(data_dir + 'motif_predictions/modisco/species-agnostic.h5', pos_patterns, neg_patterns)
