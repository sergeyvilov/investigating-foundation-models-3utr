from helpers import misc
import torch
from torch.utils.data import IterableDataset
import pysam
import numpy as np

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

                tokenized_seq = self.tokenizer(seq, add_special_tokens=False)['input_ids']

                #N_tokens_overlap=np.random.randint(low=0,high=input_params.max_overlap_tokens),

                tokenized_chunks, _ = misc.get_chunks(tokenized_seq,
                                                       N_tokens_chunk=self.max_tokens,
                                                       N_tokens_overlap=self.max_overlap_tokens,
                                                       tokenizer_cls_token_id=self.tokenizer.cls_token_id,
                                                       tokenizer_eos_token_id=self.tokenizer.sep_token_id,
                                                       tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                                                       padding=True)

                for tokenized_chunk in tokenized_chunks:

                    attention_mask = [1 if token_id!=self.tokenizer.pad_token_id else 0 for token_id in tokenized_chunk]

                    n_tokens = np.sum([1 for token_id in tokenized_chunk if token_id!=self.tokenizer.pad_token_id])-2

                    tokenized_chunk = {'input_ids':tokenized_chunk,
                                       'seq_idx':seq_idx,
                                       'token_type_ids':[0]*len(tokenized_chunk),
                                       'attention_mask':attention_mask,
                                       'n_tokens': n_tokens}

                    yield tokenized_chunk

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
                transform=None
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

                tokenized_seq = self.tokenizer(seq, add_special_tokens=False)['input_ids']

                #N_tokens_overlap=np.random.randint(low=0,high=input_params.max_overlap_tokens),

                tokenized_chunks, _ = misc.get_chunks(tokenized_seq,
                                                       N_tokens_chunk=self.max_tokens,
                                                       N_tokens_overlap=self.max_overlap_tokens,
                                                       tokenizer_cls_token_id=self.tokenizer.cls_token_id,
                                                       tokenizer_eos_token_id=self.tokenizer.sep_token_id,
                                                       tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                                                       padding=True)

                for tokenized_chunk in tokenized_chunks:

                    attention_mask = [1 if token_id!=self.tokenizer.pad_token_id else 0 for token_id in tokenized_chunk]

                    n_tokens = np.sum([1 for token_id in tokenized_chunk if token_id!=self.tokenizer.pad_token_id])-2

                    tokenized_chunk = {'input_ids':tokenized_chunk,
                                       'seq_idx':seq_idx,
                                       'token_type_ids':[0]*len(tokenized_chunk),
                                       'attention_mask':attention_mask,
                                       'n_tokens': n_tokens}

                    yield tokenized_chunk
