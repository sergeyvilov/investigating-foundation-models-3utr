#!/usr/bin/env python

import numpy as np
import pandas as pd

import pickle
import os
import gc

import pysam

import torch
from torch.utils.data import DataLoader, IterableDataset

import argparse

from encoding_utils import sequence_encoders

import helpers.train_eval as train_eval    #train and evaluation
import helpers.misc as misc                #miscellaneous functions
from helpers.metrics import MaskedAccuracy

from models.spec_dss import DSSResNet, DSSResNetEmb, SpecAdd

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "FASTA file", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--checkpoint_dir", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--optimizer_weight", help = "initialization weight of the optimizer, use only to resume training", type = str, default = None, required = False)
parser.add_argument("--val_fraction", help = "fraction of validation dataset to use", type = float, default = 0.1, required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int,  default = 1, required = False)
parser.add_argument("--test", help = "model to inference mode", action='store_true', default = False, required = False)
parser.add_argument("--mask_at_test", help = "mask at test", type=misc.str2bool, default = True, required = False)
parser.add_argument("--species_aware", help = "use a species aware version", action='store_true', default = False, required = False)
parser.add_argument("--save_probs", help = "save probs and embeddings at test", action='store_true', default = False, required = False)
parser.add_argument("--seq_len", help = "max UTR chunk length", type = int, default = 5000, required = False)
parser.add_argument("--overlap_bp", help = "overlap between consecutive chunks", type = int, default = 128, required = False)
parser.add_argument("--mlm_probability", help = "masking probability", type = float, default = 0.15, required = False)
parser.add_argument("--train_splits", help = "split each epoch into N epochs", type = int, default = 4, required = False)
parser.add_argument("--tot_epochs", help = "total number of training epochs, (after splitting)", type = int, default = 11, required = False)
parser.add_argument("--d_model", help = "model dimensions", type = int, default = 128, required = False)
parser.add_argument("--n_layers", help = "number of layers", type = int, default = 4, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 64, required = False)
parser.add_argument("--learning_rate", help = "learning rate", type = float, default = 1e-4, required = False)
parser.add_argument("--dropout", help = "model dropout", type = float, default = 0., required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0., required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = str, default = [], required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

input_params.save_at = misc.list2range(input_params.save_at)

for param_name in ['output_dir', '\\',
'fasta',   '\\',
'species_aware', '\\',
'test', 'mask_at_test', 'save_probs', '\\',
'seq_len', 'overlap_bp', 'mlm_probability', '\\',
'tot_epochs', 'save_at', 'train_splits', '\\',
'val_fraction', 'validate_every', '\\',
'd_model', 'n_layers', 'dropout', '\\',
'checkpoint_dir', 'optimizer_weight', '\\',
'batch_size', 'learning_rate', 'weight_decay', '\\',
]:

    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')

class SeqDataset(IterableDataset):

    def __init__(self, fasta_fa, seq_df, transform, chunk_len=1000000000, overlap_bp=0):

        if fasta_fa:
            self.fasta = pysam.FastaFile(fasta_fa)
        else:
             self.fasta = None

        self.transform = transform
        self.seq_df = seq_df

        self.start = 0
        self.end = len(self.seq_df)

        self.chunk_len = chunk_len
        self.overlap_bp = overlap_bp

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

            species_label = self.seq_df.iloc[seq_idx].species_label

            seq = seq.replace('-','')

            chunks, left_shift_last_chunk = misc.get_chunks(seq, self.chunk_len, self.overlap_bp)

            for chunk_idx,seq_chunk in enumerate(chunks):

                masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq_chunk, motifs = {})

                masked_sequence = (masked_sequence, species_label)

                chunk_meta = {'seq_name':self.seq_df.iloc[seq_idx].seq_name,
                             'seq':seq_chunk,
                             'seq_idx':seq_idx,
                             'left_shift':left_shift_last_chunk if chunk_idx==len(chunks)-1 else 0}

                yield masked_sequence, target_labels_masked, target_labels, chunk_meta


    def close(self):
        self.fasta.close()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'\nCUDA device: {torch.cuda.get_device_name(0)}\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')

gc.collect()
torch.cuda.empty_cache()

seq_df = pd.read_csv(input_params.fasta + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])

seq_df['species_name'] = seq_df.seq_name.apply(lambda x:x.split(':')).apply(lambda x:x[1] if len(x)>1 else 'Homo_sapiens')

all_species = sorted(seq_df.species_name.unique())

if input_params.species_aware:
    species_encoding = {species:idx for idx,species in enumerate(all_species)}
else:
    species_encoding = {species:0 for species in all_species}

seq_df['species_label'] = seq_df.species_name.map(species_encoding)

#seq_df = seq_df.sample(frac = 1., random_state = 1) #DO NOT SHUFFLE, otherwise too slow


if not input_params.test:

    #Train and Validate

    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.seq_len, total_len = input_params.seq_len,
                                                      mask_rate = input_params.mlm_probability, split_mask = True)

    N_train = int(len(seq_df)*(1-input_params.val_fraction))
    train_df, test_df = seq_df.iloc[:N_train], seq_df.iloc[N_train:]

    train_fold = np.repeat(list(range(input_params.train_splits)),repeats = N_train // input_params.train_splits + 1 )
    train_df['train_fold'] = train_fold[:N_train]

    train_dataset = SeqDataset(input_params.fasta, train_df, transform = seq_transform, chunk_len=input_params.seq_len, overlap_bp=input_params.overlap_bp)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size, num_workers = 1, worker_init_fn=misc.worker_init_fn, collate_fn = None, shuffle = False)

    test_dataset = SeqDataset(input_params.fasta, test_df, transform = seq_transform, chunk_len=input_params.seq_len)

    test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size, num_workers = 1, worker_init_fn=misc.worker_init_fn, collate_fn = None, shuffle = False)

elif input_params.save_probs:

    if input_params.mask_at_test:
        seq_transform = sequence_encoders.RollingMasker(mask_stride = 50, frame = 0)
    else:
        seq_transform = sequence_encoders.PlainOneHot(frame = 0, padding = 'none')

    test_dataset = SeqDataset(input_params.fasta, seq_df, transform = seq_transform, chunk_len=input_params.seq_len)

    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 1, collate_fn = None, worker_init_fn=misc.worker_init_fn, shuffle = False)

else:

    #Test

    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.seq_len, total_len = input_params.seq_len,
                                                      mask_rate = input_params.mlm_probability, split_mask = True, frame = 0)

    test_dataset = SeqDataset(input_params.fasta, seq_df, transform = seq_transform)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size, num_workers = 1, worker_init_fn=misc.worker_init_fn, collate_fn = None, shuffle = False)

species_encoder = SpecAdd(embed = True, encoder = 'label', d_model = input_params.d_model)

model = DSSResNetEmb(d_input = 5, d_output = 5, d_model = input_params.d_model, n_layers = input_params.n_layers,
                     dropout = input_params.dropout, embed_before = True, species_encoder = species_encoder)

model = model.to(device)

model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, lr = input_params.learning_rate, weight_decay = input_params.weight_decay)

last_epoch = 0

if input_params.checkpoint_dir:

    if torch.cuda.is_available():
        #load on gpu
        model.load_state_dict(torch.load(input_params.checkpoint_dir + '/model.pt'))
        print('model weights loaded')
        if os.path.isfile(input_params.checkpoint_dir + '/optimizer.pt'):
            optimizer.load_state_dict(torch.load(input_params.checkpoint_dir + '/optimizer.pt'))
            print('optimizer loaded')

    else:
        #load on cpu
        model.load_state_dict(torch.load(input_params.checkpoint_dir + '/model.pt', map_location=torch.device('cpu')))
        print('model weights loaded')
        if os.path.isfile(input_params.checkpoint_dir + '/optimizer.pt'):
            optimizer.load_state_dict(torch.load(input_params.checkpoint_dir + '/optimizer.pt', map_location=torch.device('cpu')))
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

        train_dataset.seq_df = train_df[train_df.train_fold == (epoch-1) % input_params.train_splits]
        train_dataset.end = len(train_dataset.seq_df)

        print(f'using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, device,
                            silent = True)

        print(f'epoch {epoch} - train, {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at or -1 in input_params.save_at: #save model weights

            misc.save_model_weights(model, optimizer, scheduler=None, output_dir=weights_dir, epoch=epoch, save_at=input_params.save_at)

        if input_params.val_fraction>0 and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics, *_ =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                    silent = True)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')

else:

    print(f'EPOCH {last_epoch}: Test/Inference...')

    test_metrics, test_probs, motif_probas =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                                                          save_probs = input_params.save_probs,
                                                          silent = True)



    print(f'epoch {last_epoch} - test, {metrics_to_str(test_metrics)}')

    if input_params.save_probs:

        os.makedirs(input_params.output_dir, exist_ok = True)

        with open(input_params.output_dir + '/predictions.pickle', 'wb') as f:
            seq_names, seqs, embeddings, probs, losses = zip(*test_probs)
            pickle.dump({'seq_names':seq_names, 'seqs':seqs, 'embeddings':embeddings, 'probs':probs, 'losses':losses, 'fasta':input_params.fasta},f)


print()
print(f'peak GPU memory allocation: {round(torch.cuda.max_memory_allocated(device)/1024/1024)} Mb')
print('Done')
