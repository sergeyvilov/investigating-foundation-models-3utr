#!/usr/bin/env python

import numpy as np
import pandas as pd

#from glob import glob
import os

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/whole_genome/'

output_dir = data_dir + 'shuffled_splits/'

species_index = data_dir + 'all_species.fa.fai' #index of the combined FASTA (all species)

CHUNK_LEN = 100_000 #length of contigs in which long sequences will be split
OVERLAP_BP = 50 # overlap between subsequent contigs

N_SPLITS = 1000 #total number of splits, to be used in a slurm array

seqs = pd.read_csv(species_index,sep = '\t',
                       header = None,
                       usecols = [0,1],
                       names = ['seq_name','seq_len'],
                       index_col = 0).squeeze()

contigs = []

for seq_name, seq_len in seqs.items():
    for start in range(1,max(seq_len-OVERLAP_BP,1),CHUNK_LEN-OVERLAP_BP):
        end = min(start+CHUNK_LEN-1,seq_len)
        if end-start+1>=6:
            contigs.append(f'{seq_name}:{start}-{end}')

np.random.seed(42)
np.random.shuffle(contigs)

os.makedirs(output_dir, exist_ok=True)

n_contigs = len(contigs)
contigs_per_split = n_contigs//N_SPLITS+1

for file_idx, contig_start_idx in enumerate(range(0,n_contigs,contigs_per_split)):
    with open(output_dir +f'/split_{file_idx}.txt','w') as f:
        for contig in contigs[contig_start_idx:contig_start_idx+contigs_per_split]:
            f.write(f'{contig}\n')
