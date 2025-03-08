from helpers import misc
import torch
from torch.utils.data import IterableDataset
import pysam
import numpy as np

class TextDataset(IterableDataset):

    def __init__(self,
                dataset_txt,        #txt file with sequence in each line
                transform,
                max_tokens,         #maximum tokens in chunk
                max_overlap_tokens, #maximum overlap between chunks
                species_tokens=None,  #list of species for all sequences
                start_seq_idx=0,    #sequences to skip
                size=None,          #total number of sequences
                rank=0,             #rank for parallel training
                world_size=1,       #world size for parallel training
                ):


        self.dataset = dataset_txt
        self.transform = transform
        self.species_tokens = species_tokens
        self.start_seq_idx = start_seq_idx
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.max_overlap_tokens = max_overlap_tokens

    def __len__(self):
        return self.size

    def __iter__(self):

        #worker_total_num = torch.utils.data.get_worker_info().num_workers
        #worker_id = torch.utils.data.get_worker_info().id

        with open(self.dataset, 'r') as f:

            for seq_idx, seq in enumerate(f):

                if seq_idx<self.start_seq_idx:
                    continue

                if seq_idx%self.world_size!=self.rank:
                    continue

                seq = seq.replace('-','')

                tokenized_chunks, left_shift_last_chunk = misc.get_chunks(seq,
                                                       N_tokens_chunk=self.max_tokens,
                                                       N_tokens_overlap=self.max_overlap_tokens,
                                                       tokenizer_cls_token_id=None,
                                                       tokenizer_eos_token_id=None,
                                                       tokenizer_pad_token_id=None,
                                                       padding=False)

                species_label = self.species_tokens[seq_idx] if self.species_tokens is not None else 0

                for chunk_idx,seq_chunk in enumerate(tokenized_chunks):

                    masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq_chunk.upper(), motifs = {})

                    chunk = {'input_ids':masked_sequence,
                                 'labels': target_labels_masked,
                                 'labels_unmasked':target_labels,
                                 'species_label':species_label,
                                 'seq':seq_chunk,
                                 'seq_idx':seq_idx,
                                 'n_tokens':len(seq_chunk),
                                 'left_shift':left_shift_last_chunk if chunk_idx==len(tokenized_chunks)-1 else 0}

                    yield chunk


class FASTADataset(IterableDataset):

    def __init__(self,
                fasta_fa,          #FASTA file
                transform,
                seqs_list,          #list of sequence names in FASTA file
                max_tokens,         #maximum tokens in chunk
                max_overlap_tokens, #maximum overlap between chunks
                species_tokens=None,  #list of species for all sequences
                start_seq_idx=0,    #sequences to skip
                size=None,          #total number of sequences
                rank=0,             #rank for parallel training
                world_size=1,       #world size for parallel training
                ):


        self.fasta_fa = fasta_fa
        self.transform = transform
        self.start_seq_idx = start_seq_idx
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.max_overlap_tokens = max_overlap_tokens
        self.seqs_list = seqs_list
        self.species_tokens = species_tokens

    def __len__(self):
        return self.size

    def __iter__(self):

        #worker_total_num = torch.utils.data.get_worker_info().num_workers
        #worker_id = torch.utils.data.get_worker_info().id

        with pysam.FastaFile(self.fasta_fa) as fasta:

            for seq_idx, seq_name in enumerate(self.seqs_list):

                if seq_idx<self.start_seq_idx:
                    continue

                if seq_idx%self.world_size!=self.rank:
                    continue

                seq = fasta.fetch(seq_name)

                seq = seq.replace('-','')

                tokenized_chunks, left_shift_last_chunk = misc.get_chunks(seq,
                                                       N_tokens_chunk=self.max_tokens,
                                                       N_tokens_overlap=self.max_overlap_tokens,
                                                       tokenizer_cls_token_id=None,
                                                       tokenizer_eos_token_id=None,
                                                       tokenizer_pad_token_id=None,
                                                       padding=False)

                species_label = self.species_tokens[seq_idx] if self.species_tokens is not None else 0

                for chunk_idx,seq_chunk in enumerate(tokenized_chunks):

                    masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq_chunk.upper(), motifs = {})

                    chunk = {'input_ids':masked_sequence,
                                 'labels': target_labels_masked,
                                 'labels_unmasked':target_labels,
                                 'species_label':species_label,
                                 'seq_name':seq_name,
                                 'seq':seq_chunk,
                                 'seq_idx':seq_idx,
                                 'n_tokens':len(seq_chunk),
                                 'left_shift':left_shift_last_chunk if chunk_idx==len(tokenized_chunks)-1 else 0}

                    yield chunk
