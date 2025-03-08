from helpers import misc
import torch
from torch.utils.data import IterableDataset
import pysam
import numpy as np

def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]

class TextDataset(IterableDataset):

    def __init__(self,
                dataset_txt,        #txt file with sequence in each line
                tokenizer,
                max_tokens,         #maximum tokens in chunk
                max_overlap_tokens, #maximum overlap between chunks
                start_seq_idx=0,    #sequences to skip
                size=None,          #total number of sequences
                rank=0,             #rank for parallel training
                world_size=1,       #world size for parallel training
                ):


        self.dataset = dataset_txt
        self.tokenizer = tokenizer
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

                seq = seq.upper().replace('-','')

                if len(seq)<6:
                        continue

                k_merized_seq = kmers_stride1(seq)

                tokenized_seq = self.tokenizer.encode_plus(k_merized_seq, add_special_tokens=False)['input_ids']

                tokenized_chunks, _ = misc.get_chunks(tokenized_seq,
                                                       N_tokens_chunk=self.max_tokens,
                                                       N_tokens_overlap=self.max_overlap_tokens,
                                                       tokenizer_cls_token_id=self.tokenizer.cls_token_id,
                                                       tokenizer_eos_token_id=self.tokenizer.sep_token_id,
                                                       tokenizer_pad_token_id=None,
                                                       padding=False)

                for tokenized_chunk in tokenized_chunks:
                    n_tokens = len(tokenized_chunk)-2
                    tokenized_chunk = torch.LongTensor(tokenized_chunk)
                    yield tokenized_chunk,n_tokens,seq_idx

class FASTADataset(IterableDataset):

    def __init__(self,
                fasta_fa,          #FASTA file
                tokenizer,
                seqs_list,          #list of sequence names in FASTA file
                max_tokens,         #maximum tokens in chunk
                max_overlap_tokens, #maximum overlap between chunks
                start_seq_idx=0,    #sequences to skip
                size=None,          #total number of sequences
                rank=0,             #rank for parallel training
                world_size=1,       #world size for parallel training
                species_tokens=None,
                transform=None,
                ):


        self.fasta_fa = fasta_fa
        self.tokenizer = tokenizer
        self.start_seq_idx = start_seq_idx
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.max_overlap_tokens = max_overlap_tokens
        self.seqs_list = seqs_list

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

                seq = seq.upper().replace('-','')

                if len(seq)<6:
                        continue

                k_merized_seq = kmers_stride1(seq)

                tokenized_seq = self.tokenizer.encode_plus(k_merized_seq, add_special_tokens=False)['input_ids']

                tokenized_chunks, _ = misc.get_chunks(tokenized_seq,
                                                       N_tokens_chunk=self.max_tokens,
                                                       N_tokens_overlap=self.max_overlap_tokens,
                                                       tokenizer_cls_token_id=self.tokenizer.cls_token_id,
                                                       tokenizer_eos_token_id=self.tokenizer.sep_token_id,
                                                       tokenizer_pad_token_id=None,
                                                       padding=False)

                for tokenized_chunk in tokenized_chunks:
                    n_tokens = len(tokenized_chunk)-2
                    tokenized_chunk = torch.LongTensor(tokenized_chunk)
                    yield tokenized_chunk,n_tokens,seq_idx
