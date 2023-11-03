#!/usr/bin/env python
# coding: utf-8

# Make an alignment from a MAF file

import pandas as pd
from collections import defaultdict
import sys
import warnings

#maf_file = sys.argv[1]

GAPS_TOL = 50 #1e10
MSA_BLOCK_LENGTH = 1e10 #150000

def update_msa(msa, alignment_block, human_current_pos, human_start_pos, block_len):

    '''
    Update msa after each alignment block
    '''

    updated_contigs = defaultdict(set) #sequences indices within contigs updated by the block

    for contig_name in alignment_block.keys():

        gaps_len = []

        for seq_idx, (seq, start, seq_len, orient) in enumerate(alignment_block[contig_name]): #loop over fragments in the block
            for old_seq_idx, (old_seq, old_orient, old_start, next_expected_pos) in enumerate(msa[contig_name]): #loop over already existing sequences
                gap_len = start-next_expected_pos #gap length if added to the current sequence
                if gap_len>=0 and gap_len<GAPS_TOL and old_orient==orient:
                    gaps_len.append((seq_idx, old_seq_idx, gap_len))

        gaps_len.sort(key=lambda x:x[2]) #sort by gap_len

        seq_idx_to_exclude, old_seq_idx_to_exclude = [], []

        for seq_idx, old_seq_idx, _ in gaps_len:
            #add fragments to already existing sequences starting from the minimal gap pair
            if seq_idx not in seq_idx_to_exclude and old_seq_idx not in old_seq_idx_to_exclude:

                seq, start, seq_len, _ = alignment_block[contig_name][seq_idx]
                next_expected_pos = start+seq_len
                msa[contig_name][old_seq_idx][3] = next_expected_pos #expected start position for the next fragment of this sequence
                msa[contig_name][old_seq_idx][0] += seq #extend the seqeunce with the fragment
                updated_contigs[contig_name].add(old_seq_idx) #mark the update

                seq_idx_to_exclude.append(seq_idx) #don't use this fragment anymore
                old_seq_idx_to_exclude.append(old_seq_idx) #don't add to this sequence anymore

        for seq_idx, (seq, start, seq_len, orient) in enumerate(alignment_block[contig_name]):
            if not seq_idx in seq_idx_to_exclude:
                #add new sequence
                gap_len = human_current_pos-human_start_pos #insert gaps from the beginning of the alignment to the start of the sequence
                gaps = '-'*gap_len
                next_expected_pos = start+seq_len #expected start position for the next fragment of this sequence
                seq = gaps+seq
                msa[contig_name].append([seq, orient, start, next_expected_pos]) #start new sequence with the current fragment
                updated_contigs[contig_name].add(len(msa[contig_name])-1) #mark the update

    for contig_name, subseq in msa.items():
        #add gaps to all sequences that were not in the current block
        missing_seq_idx = set(range(len(subseq)))-updated_contigs[contig_name]
        for seq_idx in missing_seq_idx:
            msa[contig_name][seq_idx][0] += '-'*block_len

def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(ch1 != ch2  for ch1, ch2 in zip(s1, s2))

def process_msa(msa, contig_lengths):

    #convert msa to a dataframe

    msa_list = []

    for full_contig_name in msa.keys():
        species_name = full_contig_name.split('.')[0] #separate species name from full contig name
        contig_name = '.'.join(full_contig_name.split('.')[1:])
        for subseq, orient, start, end in msa[full_contig_name]: #end corresponds to the next expected position for this subsequence (i.e. not inclusive)
            if orient == '-':
                #convert coordinates on the negative strand
                old_start, old_end = start, end
                start = contig_lengths[full_contig_name] - old_end
                end = contig_lengths[full_contig_name] - old_start
            msa_list.append((species_name, contig_name, start, end, subseq, orient))

    msa_df = pd.DataFrame(msa_list, columns=['species', 'contig', 'start', 'end', 'seq', 'orient'])

    #compute hamming distance btw human sequence and all other sequences, sort by hamming distance

    ref_seq = msa_df.iloc[0].seq

    msa_df['hamming'] = msa_df.seq.apply(lambda seq:hamming_distance(seq, ref_seq)) #add hamming distance between Homosapiens and all other sequences

    msa_df.iloc[1:] = msa_df.iloc[1:].sort_values(by=['species', 'hamming', 'contig']) #sort elverything except the 1st line (Homo sapiens)


    #detect overlaps

    msa_df['overlap_idx'] = [[] for row_idx in range(len(msa_df))]

    #detect overlapping subsequences within  the same contig: very rare event!
    for species in msa_df.species.unique():
        species_df = msa_df[msa_df.species == species][['contig', 'start', 'end']]
        for contig in species_df.contig.unique():
            contig_coords = species_df[species_df.contig == contig][['start', 'end']].reset_index().values
            for idx_s1, start_s1, end_s1 in contig_coords:
                for idx_s2, start_s2, end_s2 in contig_coords:
                    if idx_s2!=idx_s1:
                        if ((start_s1>=start_s2 and start_s1<end_s2) or (start_s2>=start_s1 and start_s2<end_s1)
                            or (end_s1>start_s2 and end_s1<=end_s2) or (end_s2>start_s1 and end_s2<=end_s1)):
                            msa_df.loc[idx_s1, 'overlap_idx'].append(idx_s2)
                            msa_df.loc[idx_s2, 'overlap_idx'].append(idx_s1)
                            warning = f'Overlap detected {species}: {contig}'
                            warnings.warn(warning)

    unique_seq = {}

    #construct a single sequence per species by performing netting
    #we choose the sequence with the smallest hamming distance as the reference
    #we fill gaps in the reference sequence using sequences with larger and larger hamming distance
    for species in msa_df.species.unique():
            #idx_to_exclude = []
            species_df = msa_df[msa_df.species == species][['seq','overlap_idx']]
            species_seq = list(species_df.iloc[0].seq)
            idx_to_exclude = list(species_df.iloc[0].overlap_idx) #we won't consider any sequences that overlap the reference
            for idx, (seq, overlap_idx) in species_df.iloc[1:].iterrows():
                if idx not in idx_to_exclude:
                    idx_to_exclude.extend(overlap_idx) #we won't consider any sequences that overlap the current sequence
                    species_seq = [species_seq[i] if species_seq[i]!='-' else seq[i] for i in range(len(species_seq))] #fill gaps in the refseq using current sequence
            species_seq = ''.join(species_seq)
            #unique_seq[species] = (species_seq,hamming_distance(species_seq, ref_seq))
            unique_seq[species] = species_seq

    return unique_seq

###################################################################################################################

msa_blocks = [] #each msa block is processed independently, max MSA_BLOCK_LENGTH bases per block

current_msa_block = defaultdict(list) #initialize current msa block

current_alignment_block = defaultdict(list) #initialize current alignment block

contig_lengths = dict() #full contig length, ro recalculate coords on negative strand

if len(sys.argv)>1:
    input_f = open(sys.argv[1], 'r') #if input file is given as an argument
else:
    input_f = sys.stdin #else read from pipe

try:
    for line in input_f:

        if line.startswith('s'):

            _, contig_name, start, seq_len, orient, contig_length, seq = line.split()

            contig_lengths[contig_name] = int(contig_length) #full length of current contig

            start, seq_len = int(start), int(seq_len)

            if  contig_name.startswith('Homo_sapiens'): #first sequence in the alignment block=reference sequence (Homo sapiens)
                human_current_pos = start #human start pos for the current block
                current_alignment_block_len = len(seq) #width of the current block
                if not current_msa_block:
                    msa_block_human_start_pos = human_current_pos #start pos in the independent msa block
                    if not msa_blocks:
                        maf_human_start_pos = human_current_pos #if that's the first alignment block in the MAF file
                        human_contig_name = contig_name

            seq = seq.upper() #repeats to uppercase

            current_alignment_block[contig_name].append((seq, start, seq_len, orient)) #(sequence, start pos for this sequence in the block, sequence length, orientation)

        elif current_alignment_block and line=='\n': #empty line after each alignment block
            #process previous block
            update_msa(current_msa_block, current_alignment_block, human_current_pos, msa_block_human_start_pos, current_alignment_block_len)
            current_alignment_block = defaultdict(list) #empty current block

            if human_current_pos-msa_block_human_start_pos>=MSA_BLOCK_LENGTH:
                msa = process_msa(current_msa_block, contig_lengths)
                msa_blocks.append(msa)
                current_msa_block = defaultdict(list)

    if current_msa_block:
        msa = process_msa(current_msa_block, contig_lengths)
        msa_blocks.append(msa)

finally:

    input_f.close()

all_species = msa.keys()

unique_seq = defaultdict(lambda:'')

for species in all_species:
    for msa_block in msa_blocks:
        if species in msa_block.keys():
            unique_seq[species] += msa_block[species]
        else:
            unique_seq[species] += '-'*len(msa_block['Homo_sapiens'])

for species, seq in unique_seq.items():
    if species == 'Homo_sapiens':
        species =  f'{human_contig_name}/{maf_human_start_pos}-{current_msa_block[human_contig_name][0][-1]}'
    seq = ''.join([s if s in 'ACTG' else '-' for s in seq])
    print('>'+species)
    print(seq)
