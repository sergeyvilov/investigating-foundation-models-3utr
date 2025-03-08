import os
import gc
import pickle
import argparse
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

import pysam

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from transformers import BertConfig, BertForMaskedLM

import helpers.misc as misc                #miscellaneous functions
import helpers.train_eval as train_eval    #train and evaluation
from DNABERT.src.transformers.tokenization_dna import DNATokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'\nCUDA device: {torch.cuda.get_device_name(0)}\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')


def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]

class SeqDataset(IterableDataset):

    def __init__(self, fasta_fa, seq_df):

        if fasta_fa:
            self.fasta = pysam.FastaFile(fasta_fa)
        else:
             self.fasta = None

        self.seq_df = seq_df
        self.start = 0
        self.end = len(self.seq_df)

    def __len__(self):
        return len(self.seq_df)

    def __iter__(self):

        #worker_total_num = torch.utils.data.get_worker_info().num_workers
        #worker_id = torch.utils.data.get_worker_info().id

        for seq_idx in range(self.start,self.end):

            if self.fasta:
                seq = self.fasta.fetch(self.seq_df.iloc[seq_idx].seq_name).upper()
            else:
                seq = self.seq_df.iloc[seq_idx].seq.upper()

            #species_label = self.seq_df.iloc[idx].species_label

            seq = seq.replace('-','')

            if len(seq)<6:
                    continue

            k_merized_seq = kmers_stride1(seq)

            tokenized_seq = tokenizer.encode_plus(k_merized_seq, add_special_tokens=False)['input_ids']

            #N_tokens_overlap=np.random.randint(low=0,high=input_params.max_overlap_tokens),

            tokenized_chunks, _ = misc.get_chunks(tokenized_seq,
                                                   N_tokens_chunk=input_params.max_tokens,
                                                   N_tokens_overlap=input_params.max_overlap_tokens,
                                                   tokenizer_cls_token_id=tokenizer.cls_token_id,
                                                   tokenizer_eos_token_id=tokenizer.sep_token_id,
                                                   tokenizer_pad_token_id=None,
                                                   padding=False)

            for tokenized_chunk in tokenized_chunks:
                tokenized_chunk = torch.LongTensor(tokenized_chunk)
                yield tokenized_chunk,seq_idx

    def close(self):
        self.fasta.close()

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "FASTA file", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--checkpoint_dir", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--val_fraction", help = "fraction of validation dataset to use", type = float, default = 0.1, required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int,  default = 16, required = False)
parser.add_argument("--species_agnostic", help = "use a species agnostic version", action='store_true', default = False, required = False)
parser.add_argument("--train_chunks", help = "split each epoch into N epochs", type = int, default = 128, required = False)
parser.add_argument("--max_overlap_tokens", help = "maximal tokenized chunk overlap", type = int, default = 50, required = False)
parser.add_argument("--max_tokens", help = "maximal chunk length in tokens", type = int, default = 512, required = False)
parser.add_argument("--step_size_up", help = "number of iterations for lr ascending", type = int, default = 10000, required = False)
parser.add_argument("--step_size_down", help = "number of iterations for lr descending", type = int, default = 200000, required = False)
parser.add_argument("--mlm_probability", help = "masking probability", type = float, default = 0.15, required = False)
parser.add_argument("--tot_epochs", help = "total number of training epochs, (after splitting)", type = int, default = 20, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 64, required = False)
parser.add_argument("--max_lr", help = "learning rate", type = float, default = 4e-4, required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0.01, required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = str, default = ['1:30'], required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

for param_name in ['output_dir', '\\',
        'fasta', 'checkpoint_dir', '\\',
        'species_agnostic', '\\',
        'tot_epochs', 'save_at', '\\',
        'train_chunks','val_fraction', 'validate_every', '\\',
        'max_tokens', 'max_overlap_tokens', 'mlm_probability', '\\',
        'batch_size', 'max_lr', 'weight_decay','step_size_up', 'step_size_down', '\\',
        ]:

            if param_name == '\\':
                print()
            else:
                print(f'{param_name.upper()}: {input_params[param_name]}')

input_params.save_at = misc.list2range(input_params.save_at)

seq_df = pd.read_csv(input_params.fasta + '.fai', header=None, sep='\t', usecols=[0,1], names=['seq_name','seq_len'])

#seq_df = seq_df.iloc[:3000]

seq_df['species_name'] = seq_df.seq_name.apply(lambda x:x.split(':')).apply(lambda x:x[1] if len(x)==2 else 'Homo_sapiens')

all_species = sorted(seq_df.species_name.unique())

if not input_params.species_agnostic:
    species_encoding = {species:idx for idx,species in enumerate(all_species)}
else:
    species_encoding = {species:0 for species in all_species}

seq_df['species_label'] = seq_df.species_name.map(species_encoding)

#seq_df = seq_df.sample(frac = 1., random_state = 1) #DO NOT SHUFFLE, otherwise too slow

tokenizer = DNATokenizer(vocab_file='./DNABERT/src/transformers/dnabert-config/bert-config-6/vocab.txt',
                        max_len=input_params.max_tokens)

config = BertConfig.from_pretrained('./DNABERT/src/transformers/dnabert-config/bert-config-6/config.json')

model = BertForMaskedLM(config).to(device)

def collate_fn(batch):
    examples, seq_idx = zip(*batch)
    seq_padded = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    return misc.mask_tokens(seq_padded, tokenizer, input_params.mlm_probability), seq_idx

N_train = int(len(seq_df)*(1-input_params.val_fraction))
train_df, test_df = seq_df.iloc[:N_train], seq_df.iloc[N_train:]

train_chunk = np.repeat(list(range(input_params.train_chunks)),repeats = N_train // input_params.train_chunks + 1 )
train_df['train_chunk'] = train_chunk[:N_train]

train_dataset = SeqDataset(input_params.fasta, train_df)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size,
                              num_workers = 1, worker_init_fn=misc.worker_init_fn, collate_fn = collate_fn, shuffle = False)

test_dataset = SeqDataset(input_params.fasta, test_df)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size,
                             num_workers = 1,  worker_init_fn=misc.worker_init_fn, collate_fn = collate_fn, shuffle = False)

gc.collect()
torch.cuda.empty_cache()

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": input_params.weight_decay,
    },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=input_params.max_lr, eps=1e-6, betas=(0.9,0.98))

#scheduler = get_linear_schedule_with_warmup(
#        optimizer, num_warmup_steps=input_params.step_size_up, num_training_steps=200000
#    )

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0, max_lr = input_params.max_lr,
                                             step_size_up = input_params.step_size_up, step_size_down = input_params.step_size_down, cycle_momentum=False)


last_epoch = 0

if input_params.checkpoint_dir:

    model = BertForMaskedLM.from_pretrained(input_params.checkpoint_dir).to(device)
    print('model loaded')

    if os.path.isfile(input_params.checkpoint_dir + '/optimizer.pt'):
            optimizer.load_state_dict(torch.load(input_params.checkpoint_dir + '/optimizer.pt'))
            scheduler.load_state_dict(torch.load(input_params.checkpoint_dir + '/scheduler.pt'))
            print('optimizer loaded')

    last_epoch = int(input_params.checkpoint_dir.rstrip('/').split('_')[-1]) #infer previous epoch from input_params.checkpoint_dir

weights_dir = os.path.join(input_params.output_dir, 'checkpoints') #dir to save model weights at save_at epochs

if input_params.save_at:
    os.makedirs(weights_dir, exist_ok = True)

def metrics_to_str(metrics):
    loss, total_acc, masked_acc = metrics
    return f'loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}'

from helpers.misc import print    #print function that displays time

if not input_params.test:

    for epoch in range(last_epoch+1, input_params.tot_epochs+1):

        print(f'EPOCH {epoch}: Training...')

        train_dataset.seq_df = train_df[train_df.train_chunk == (epoch-1) % input_params.train_chunks]
        train_dataset.end = len(train_dataset.seq_df)

        print(f'using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, device, scheduler=scheduler,
                            silent = True)

        print(f'epoch {epoch} - train ({scheduler.last_epoch+1} iterations), {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at or -1 in input_params.save_at: #save model weights

            misc.save_model_weights(model, optimizer, scheduler, weights_dir, epoch, input_params.save_at)

        if input_params.val_fraction>0 and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                    silent = True)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')



# In[ ]:
