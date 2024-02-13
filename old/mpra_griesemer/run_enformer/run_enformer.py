#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pysam
import tensorflow.compat.v2 as tf
import os
from collections import defaultdict
import pickle
import sys


# In[2]:


#import tensorflow_hub as hub
#enformer_model = hub.load("https://tfhub.dev/deepmind/enformer/1").model


# In[3]:


SEQ_LENGTH = 393216 #Enformer input sequences length
N_bins = 896 #Number of Enformer output bins


# In[4]:


start_row = int(sys.argv[1])
stop_row = int(sys.argv[2])


# In[5]:


print(f'Start row: {start_row}')
print(f'Stop row: {stop_row}')


# In[6]:


enformer_model_dir = '/s/project/mll/sergey/effect_prediction/tools/enformer/model/'
fasta_fa = '/s/project/mll/sergey/ref_genomes/hg38.fa.gz'


# In[7]:


datadir = '/s/project/mll/sergey/effect_prediction/MLM/'


# In[8]:


variants_tsv = datadir + 'griesemer/GRCh38_UTR_variants.tsv' #all positions are 0-based [start, end)
output_dir = datadir + 'griesemer/enformer/predictions/'


# In[9]:


#targets_idx = np.array(np.arange(0,674)) #DNASE


# In[10]:


variants_df = pd.read_csv(variants_tsv, sep='\t')

variants_df = variants_df[variants_df.other_var_in_oligo_window.isna()] #seeding multiple variants into oligo sequence isn't currently supported

variants_df['vartype'] = variants_df.apply(lambda x: 'SNP' if len(x.ref)==len(x.alt) else
                                            'DEL' if len(x.ref)>len(x.alt) else 'INS', axis=1)

variants_df = variants_df.sort_values(by='oligo_id')

variants_df = variants_df[variants_df.tag == 'alt'] #take only one row pro variant (ref or alt)

#change ref and alt for non-standard delection
variants_df.loc[variants_df.mpra_variant_id=='chr4.4:56297874i','ref'] = 'CAG'
variants_df.loc[variants_df.mpra_variant_id=='chr4.4:56297874i', 'alt'] = 'C'

variants_df = variants_df.iloc[start_row:stop_row]


# In[11]:


def check_ref(seq, variant, varpos):
    '''
    Detect reference mismatches
    '''
    if variant.vartype != 'DEL' and seq[varpos] != variant.ref:
        return False
    elif variant.vartype == 'DEL' and ''.join(seq[varpos:varpos+len(variant.ref)]) != variant.ref:
        return False
    
    return True

def insert_variant(ref, alt, pos, seq, seq_pos):
    '''
    insert a variant into an existing sequence
    seq - array of 'A', 'T', 'C', 'G' or 'N'
    seq_pos - absolute positions of sequence bp in the genome
    '''
    varpos = seq_pos.index(pos) #index inside the sequence of variant position (relative position)
    if len(alt)==len(ref):
        assert seq[varpos]==ref, 'Wrong reference allele'
        seq[varpos] = alt
    elif len(alt)>len(ref): #insertion
        assert seq[varpos]==ref, 'Wrong reference allele'
        seq = seq[:varpos] + list(alt) + seq[varpos+1:]
        seq_pos = seq_pos[:varpos] + [seq_pos[varpos]]*len(alt) + seq_pos[varpos+1:] #assign all inserted bases the same position
    else: #deletion
        assert seq[varpos:varpos+len(ref)]==list(ref), 'Wrong reference allele'
        seq = seq[:varpos+1] + seq[varpos+len(ref):]
        seq_pos = seq_pos[:varpos+1] + seq_pos[varpos+len(ref):]
    return seq, seq_pos

def center_around_tss(seq, seq_pos, tss_pos):
    '''
    center the sequence around the TSS
    seq - array of 'A', 'T', 'C', 'G' or 'N'
    seq_pos - absolute positions of sequence bp in the genome
    tss_pos - absolute position of TSS in the genome
    '''

    centered_seq = ['N']*SEQ_LENGTH #initialize centered sequence

    tss_idx = seq_pos.index(tss_pos) #TSS index in the input sequence

    left_seq = seq[max(0,tss_idx-SEQ_LENGTH//2):tss_idx] #part of the input sequence to the left of TSS
    right_seq = seq[tss_idx:tss_idx+SEQ_LENGTH//2] #part of the input sequence to the right of TSS
    
    #insert left and right parts of the input sequence to the centered sequence
    centered_seq[SEQ_LENGTH//2:SEQ_LENGTH//2+len(right_seq)] =  right_seq
    centered_seq[SEQ_LENGTH//2-len(left_seq):SEQ_LENGTH//2] = left_seq

    return centered_seq

def reverse_complement(seq):
    '''
    reverse complement of a given sequence
    '''
    s = list(map(lambda x:{'A':'T','C':'G','T':'A','G':'C'}.get(x,'N'),seq))
    return s[::-1]

def roll_seq(seq, shift):
    '''
    shift a sequence to right (positive shift) or to left (negative shift)
    pad with 'N'
    '''
    if shift>0:
        return ['N']*shift + seq[:-shift]
    else:
        return seq[-shift:] + ['N']*(-shift)
    
def one_hot(seq):
    '''
    One-hot encoding in order 'ACGT'
    '''
    seq = np.array(seq)
    s = np.vstack((seq=='A',seq=='C',seq=='G',seq=='T')).astype(int).T
    return np.expand_dims(s,0)

def enformer_predict(refseq_c, altseq_c):
    '''
    get enformer predictions for centered reference and alternative sequences
    '''
    #all_pred = []
    
    sequences = []
    for seq in refseq_c, reverse_complement(refseq_c), altseq_c, reverse_complement(altseq_c):
        for subseq in one_hot(seq), one_hot(roll_seq(seq,47)),one_hot(roll_seq(seq,-47)): 
            sequences.append(subseq[0,:])
            #pred = enformer_model.predict_on_batch(subseq)['human'].numpy()
            #all_pred.append(pred[:,N_bins//2,:]) #only the central bin

    #all_pred = np.vstack(all_pred)
    
    all_pred = enformer_model.predict_on_batch(sequences)['human'].numpy()
    
    all_pred = all_pred[:,N_bins//2,:]#only the central bin

    ref_pred = all_pred[:6,:].mean(axis=0) #average for seq, shifted seq (right), shifted seq (left) and reverse complement 
    alt_pred = all_pred[6:,:].mean(axis=0)

    #log2fc = np.log2(alt_pred[targets_idx]/ref_pred[targets_idx]).mean()
    
    return ref_pred, alt_pred


# In[12]:


os.makedirs(output_dir, exist_ok=True)

fasta = pysam.FastaFile(fasta_fa)

enformer_model = tf.keras.models.load_model(enformer_model_dir).model


# In[13]:


n_mismatches = 0

enformer_preds = {}

ref_pred, alt_pred = None,None

row_idx = 1

for var_idx, variant in variants_df.iterrows():
    
    print(f'predicting for variant {var_idx}: {variant.mpra_variant_id} ({row_idx}/{len(variants_df)})')
    
    refseq = fasta.fetch(variant.chrom, max(variant.var_start-SEQ_LENGTH//2,0), variant.var_start+SEQ_LENGTH//2) #fetch a region of SEQ_LENGTH around the variant
    
    refseq = list(refseq.upper())
    
    refseq_left_pos = max(int(variant.var_start-SEQ_LENGTH//2),0) #actual absolute left position in the fetched sequence
    refseq_right_pos = refseq_left_pos+len(refseq) #actual absolute right position in the fetched sequence
    
    refseq_pos = list(range(refseq_left_pos,refseq_right_pos)) #all absolute positions in refseq
    
    varpos = int(variant.var_start - refseq_left_pos) #relative variant position in the sequence

    if not check_ref(refseq, variant, varpos):
        #check if reference allele is correct
        print('Wrong reference allele!')
        n_mismatches += 1
        continue
    
    altseq, altseq_pos = list(refseq), list(refseq_pos)
    
    altseq, altseq_pos = insert_variant(variant.ref, variant.alt, variant.var_start, altseq, altseq_pos)
    
    refseq_c = center_around_tss(refseq, refseq_pos, variant.var_start) #center around variant position
    altseq_c = center_around_tss(altseq, altseq_pos, variant.var_start) #center around variant position
    
    #assert ''.join(refseq_c[SEQ_LENGTH//2:SEQ_LENGTH//2+len(variant.ref)])==variant.ref
    #assert ''.join(altseq_c[SEQ_LENGTH//2:SEQ_LENGTH//2+len(variant.alt)])==variant.alt
    
    ref_pred, alt_pred =  enformer_predict(refseq_c, altseq_c)
        
    enformer_preds[(variant.mpra_variant_id,'ref')] = ref_pred
    enformer_preds[(variant.mpra_variant_id,'alt')] = alt_pred

    row_idx += 1

print(f'{n_mismatches} reference mismatches detected')

with open(output_dir + f'{start_row}-{stop_row}.pickle', 'wb') as f:
    pickle.dump(enformer_preds, f)


# In[ ]:




