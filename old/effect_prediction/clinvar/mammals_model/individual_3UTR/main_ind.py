#!/usr/bin/env python

import numpy as np
import pandas as pd

import pickle
import os
import gc

import pysam

import torch
from torch.utils.data import DataLoader, Dataset

import argparse

from encoding_utils import sequence_encoders

import helpers.train_eval as train_eval    #train and evaluation
import helpers.misc as misc                #miscellaneous functions
from helpers.metrics import MaskedAccuracy

from models.spec_dss import DSSResNet, DSSResNetEmb, SpecAdd

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "FASTA file", type = str, required = True)
parser.add_argument("--species_list", help = "species list for integer encoding", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--model_weight", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--optimizer_weight", help = "initialization weight of the optimizer, use only to resume training", type = str, default = None, required = False)
parser.add_argument("--val_fraction", help = "fraction of validation dataset to use", type = float, default = 0.1, required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int,  default = 1, required = False)
parser.add_argument("--test", help = "model to inference mode", action='store_true', default = False, required = False)
parser.add_argument("--species_agnostic", help = "use a pecies agnostic version", action='store_true', default = False, required = False)
parser.add_argument("--get_embeddings", help = "save embeddings at test", action='store_true', default = False, required = False)
parser.add_argument("--get_motif_acc", help = "get motif accuracy at test", action='store_true', default = False, required = False)
parser.add_argument("--seq_len", help = "max UTR sequence length", type = int, default = 5000, required = False)
parser.add_argument("--train_splits", help = "split each epoch into N epochs", type = int, default = 4, required = False)
parser.add_argument("--tot_epochs", help = "total number of training epochs, (after splitting)", type = int, default = 11, required = False)
parser.add_argument("--d_model", help = "model dimensions", type = int, default = 128, required = False)
parser.add_argument("--n_layers", help = "number of layers", type = int, default = 4, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 128, required = False)
parser.add_argument("--learning_rate", help = "learning rate", type = float, default = 1e-4, required = False)
parser.add_argument("--dropout", help = "model dropout", type = float, default = 0., required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0., required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = str, default = [], required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

input_params.save_at = misc.list2range(input_params.save_at)

for param_name in ['output_dir', '\\',
'fasta', 'species_list', 'species_agnostic', '\\',
'test', 'get_embeddings', 'get_motif_acc', '\\',
'seq_len', '\\',
'tot_epochs', 'save_at', 'train_splits', '\\',
'val_fraction', 'validate_every', '\\',
'd_model', 'n_layers', 'dropout', '\\',               
'model_weight', 'optimizer_weight', '\\',
'batch_size', 'learning_rate', 'weight_decay', '\\',
]:

    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')

data_dir = '/s/project/mll/sergey/effect_prediction/MLM/'

motifs = pd.read_excel(data_dir + 'dominguez_2018/1-s2.0-S1097276518303514-mmc4.xlsx',
             sheet_name =1)

motifs = motifs.iloc[1:,0::2].values.flatten() #select all motifs

motifs_table = set(filter(lambda v: v==v, motifs)) #remove NaNs

random_motifs = {'ACTCC', 'ACTTA', 'ATGTC', 'CCACA', 'TGACT', 'TTCCG', 'TTGGG', 'GTGTA', 'ACAGG', 'TCGTA'} #motifs which don't overlap with the table

selected_motifs = motifs_table | random_motifs #union

selected_motifs = {motif:motif_idx+1 for motif_idx,motif in enumerate(selected_motifs)} #{'ACCTG':1, 'GGTAA':2}

class SeqDataset(Dataset):
    
    def __init__(self, fasta_fa, seq_df, transform, motifs = {}):
        
        self.fasta = pysam.FastaFile(fasta_fa)
        
        self.seq_df = seq_df
        self.transform = transform
        self.motifs = motifs
        
    def __len__(self):
        
        return len(self.seq_df)
    
    def __getitem__(self, idx):
        
        seq = self.fasta.fetch(seq_df.iloc[idx].seq_name).upper()
                
        species_label = seq_df.iloc[idx].species_label
                
        masked_sequence, target_labels_masked, target_labels, _, motif_mask = self.transform(seq, motifs = self.motifs)
        
        masked_sequence = (masked_sequence, species_label)
        
        #motif_ranges = []
        
        #for motif in selected_motifs:
        #    for match in re.finditer(motif, seq):
        #        motif_ranges.append((match.start(),match.end()))
            
        return masked_sequence, target_labels_masked, target_labels, motif_mask, seq
    
    def close(self):
        self.fasta.close()
        
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nCUDA device: GPU\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')

gc.collect()
torch.cuda.empty_cache()

seq_df = pd.read_csv(input_params.fasta + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])
seq_df['species_name'] = seq_df.seq_name.apply(lambda x:x.split(':')[1])
species_encoding = pd.read_csv(input_params.species_list, header=None).squeeze().to_dict()

if not input_params.species_agnostic:
    species_encoding = {species:idx for idx,species in species_encoding.items()}
else:
    species_encoding = {species:0 for _,species in species_encoding.items()}
    
species_encoding['Homo_sapiens'] = species_encoding['Pan_troglodytes']
seq_df['species_label'] = seq_df.species_name.map(species_encoding)


if not input_params.test:
    
    #Train and Validate
    
    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.seq_len, total_len = input_params.seq_len, 
                                                      mask_rate = 0.15, split_mask = True)
    
    N_train = int(len(seq_df)*(1-input_params.val_fraction))       
    train_df, test_df = seq_df.iloc[:N_train], seq_df.iloc[N_train:]
                  
    train_fold = np.repeat(list(range(input_params.train_splits)),repeats = N_train // input_params.train_splits + 1 )
    train_df['train_fold'] = train_fold[:N_train]

    train_dataset = SeqDataset(input_params.fasta, train_df, transform = seq_transform)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size, num_workers = 2, collate_fn = None, shuffle = False)

    test_dataset = SeqDataset(input_params.fasta, test_df, transform = seq_transform)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size, num_workers = 2, collate_fn = None, shuffle = False)

elif input_params.get_embeddings or input_params.get_motif_acc:
    
    #Test and get sequence embeddings (MPRA)
    
    seq_transform = sequence_encoders.RollingMasker(mask_stride = 50, frame = 0)

    if input_params.get_motif_acc:
        motifs = selected_motifs
    else:
        motifs = {}
        
    test_dataset = SeqDataset(input_params.fasta, seq_df, transform = seq_transform, motifs = selected_motifs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 1, collate_fn = None, shuffle = False)
    
else:
    
    #Test
    
    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.seq_len, total_len = input_params.seq_len, 
                                                      mask_rate = 0.15, split_mask = True, frame = 0)
    
    test_dataset = SeqDataset(input_params.fasta, seq_df, transform = seq_transform)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size, num_workers = 2, collate_fn = None, shuffle = False)

species_encoder = SpecAdd(embed = True, encoder = 'label', d_model = input_params.d_model)

model = DSSResNetEmb(d_input = 5, d_output = 5, d_model = input_params.d_model, n_layers = input_params.n_layers, 
                     dropout = input_params.dropout, embed_before = True, species_encoder = species_encoder)

model = model.to(device) 

model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, lr = input_params.learning_rate, weight_decay = input_params.weight_decay)

last_epoch = 0

if input_params.model_weight:

    if torch.cuda.is_available():
        #load on gpu
        model.load_state_dict(torch.load(input_params.model_weight))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight))
    else:
        #load on cpu
        model.load_state_dict(torch.load(input_params.model_weight, map_location=torch.device('cpu')))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight, map_location=torch.device('cpu')))

    last_epoch = int(input_params.model_weight.split('_')[-3]) #infer previous epoch from input_params.model_weight

predictions_dir = os.path.join(input_params.output_dir, 'predictions') #dir to save predictions
weights_dir = os.path.join(input_params.output_dir, 'weights') #dir to save model weights at save_at epochs

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
        print(f'using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, device,
                            silent = True)

        print(f'epoch {epoch} - train, {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at: #save model weights

            misc.save_model_weights(model, optimizer, weights_dir, epoch)

        if input_params.val_fraction>0 and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics, _ =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                    silent = True)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')

else:

    print(f'EPOCH {last_epoch}: Test/Inference...')

    test_metrics, test_embeddings, motif_probas =  train_eval.model_eval(model, optimizer, test_dataloader, device, 
                                                          get_embeddings = input_params.get_embeddings, 
                                                          get_motif_acc = input_params.get_motif_acc, 
                                                          selected_motifs = selected_motifs, silent = True)
    
    

    print(f'epoch {last_epoch} - test, {metrics_to_str(test_metrics)}')

    if input_params.get_embeddings:
        
        os.makedirs(input_params.output_dir, exist_ok = True)

        with open(input_params.output_dir + '/embeddings.npy', 'wb') as f:
            test_embeddings = np.vstack(test_embeddings)
            np.save(f, test_embeddings)
            
    if input_params.get_motif_acc:
        
        os.makedirs(input_params.output_dir, exist_ok = True)

        with open(input_params.output_dir + '/motif_probas.pickle', 'wb') as f:
            pickle.dump(motif_probas, f) #seq_index,motif,motif_start,avg_target_proba

        seq_df.seq_name.to_csv(input_params.output_dir + '/seq_index.csv') #save index seqeunce matchin for 1st column of motif_probas 


print()
print(f'peak GPU memory allocation: {round(torch.cuda.max_memory_allocated(device)/1024/1024)} Mb')
print('Done')
