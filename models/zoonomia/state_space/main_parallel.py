import os
import subprocess
import gc
import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from tqdm import tqdm

import random

import torch
from torch.utils.data import DataLoader

import helpers.datasets as seq_datasets
import helpers.misc as misc                #miscellaneous functions
import helpers.train_eval as train_eval    #train and evaluation

from encoding_utils import sequence_encoders
from models.spec_dss import DSSResNet, DSSResNetEmb, SpecAdd

import torch.distributed as dist

from torch.cuda.amp import GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--train_dataset", help = "txt/FASTA dataset, one sequence per line", type = str, required = False)
parser.add_argument("--val_dataset", help = "txt/FASTA dataset, one sequence per line", type = str, required = False)
parser.add_argument("--test_dataset", help = "txt/FASTA dataset, one sequence per line", type = str, required = False)
parser.add_argument("--species_list", help = "list of all species for the species-aware model", type = str, required = False)
parser.add_argument("--get_probs", help = "save probs at test", action='store_true', default = False, required = False)
parser.add_argument("--get_embeddings", help = "save embeddings at test", action='store_true', default = False, required = False)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--checkpoint_dir", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--validate_every", help = "validate every N checkpoints", type = int,  default = 0, required = False)
parser.add_argument("--species_agnostic", help = "use a species agnostic version", action='store_true', default = False, required = False)
parser.add_argument("--mixed_precision", help = "mixed precision training", action='store_true', default = False, required = False)
parser.add_argument("--steps_per_chkpt", help = "make this number of steps before the checkpoint", type = int, default = 1000, required = False)
parser.add_argument("--grad_accum_itr", help = "number of iterations for gradient accumulation", type = int, default = 1, required = False)
parser.add_argument("--max_overlap_tokens", help = "maximal tokenized chunk overlap", type = int, default = 50, required = False)
parser.add_argument("--max_tokens", help = "maximal chunk length in tokens", type = int, default = 5000, required = False)
parser.add_argument("--central_window", help = "keep predictions only for this window around the sequence center", type = int, default = None, required = False)
parser.add_argument("--predict_only_lowercase", help = "keep predictions only for the lowercased positions", action='store_true', default = None, required = False)
parser.add_argument("--step_size_up", help = "number of steps for lr ascending", type = int, default = None, required = False)
parser.add_argument("--step_size_down", help = "number of steps for lr descending", type = int, default = None, required = False)
parser.add_argument("--mlm_probability", help = "masking probability", type = float, default = 0.15, required = False)
parser.add_argument("--mask_at_test", help = "mask at test", type=misc.str2bool, default = True, required = False)
parser.add_argument("--tot_chkpt", help = "train for this number of checkpoints", type = int, default = 10000, required = False)
parser.add_argument("--batch_size", help = "batch size per single GPU", type = int, default = 64, required = False)
parser.add_argument("--d_model", help = "model dimensions", type = int, default = 128, required = False)
parser.add_argument("--n_layers", help = "number of layers", type = int, default = 4, required = False)
parser.add_argument("--dropout", help = "model dropout", type = float, default = 0., required = False)
parser.add_argument("--max_lr", help = "learning rate", type = float, default = 1e-4, required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0., required = False)
parser.add_argument("--save_at", help = "checkpoints to save model/optimizer weights, 1-based", nargs='+', type = str, default = ['1:30'], required = False)
parser.add_argument("--seed", help = "seed for reproducibility", type = int, default = None, required = False)
parser.add_argument("--whitelist", help = "include sequences only from this list", type = str, default=None, required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

torch.backends.cudnn.benchmark = True

if input_params.seed is not None:

    random.seed(input_params.seed)
    np.random.seed(input_params.seed)
    torch.manual_seed(input_params.seed)
    torch.cuda.manual_seed(input_params.seed)
    torch.cuda.manual_seed_all(input_params.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)

if "MASTER_ADDR" in os.environ and 'MASTER_PORT' in os.environ:
    DDP = True
    input_params.world_size = int(os.environ["WORLD_SIZE"])
    input_params.rank = int(os.environ['SLURM_PROCID'])
else:
    DDP = False
    input_params.world_size = 1
    input_params.rank = 0

ngpus_per_node = torch.cuda.device_count()
print(ngpus_per_node)
input_params.gpu = input_params.rank % ngpus_per_node
input_params.effective_batch_size = input_params.world_size*input_params.batch_size*input_params.grad_accum_itr

print(f'rank:{input_params.rank}, tot gpus:{ngpus_per_node}, gpu:{input_params.gpu}')

if DDP:
    dist.init_process_group(backend='nccl', world_size=input_params.world_size, rank=input_params.rank,)

if input_params.rank==0:

    for param_name in ['output_dir', '\\',
            'train_dataset','val_dataset','test_dataset','\\',
            'species_list', 'get_probs', 'get_embeddings', 'mask_at_test', '\\',
            'checkpoint_dir', '\\',
            'species_agnostic', '\\',
            'seed', '\\',
            'mixed_precision', '\\',
            'steps_per_chkpt','tot_chkpt', 'save_at', '\\',
            'validate_every', '\\',
            'max_tokens', 'max_overlap_tokens', 'mlm_probability', '\\',
            'central_window', 'predict_only_lowercase', 'whitelist', '\\',
            'd_model', 'n_layers', 'dropout', '\\',
            'world_size','batch_size', 'grad_accum_itr', 'effective_batch_size', '\\',
            'max_lr', 'weight_decay','step_size_up', 'step_size_down', '\\',
            ]:

                if param_name == '\\':
                    print()
                else:
                    print(f'{param_name.upper()}: {input_params[param_name]}')

input_params.save_at = misc.list2range(input_params.save_at)


def get_dataset(dataset_url, transform, rank=input_params.rank, world_size=input_params.world_size):
    if dataset_url.endswith('.fa'):
        #for FASTA datasets
        #datasets should be indexed with samtools
        assert os.path.getctime(dataset_url + '.fai') > os.path.getctime(dataset_url), 'Index file is older than fasta file'
        seqs = pd.read_csv(dataset_url + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name']).seq_name.values
        if input_params.whitelist:
            whitelist = pd.read_csv(input_params.whitelist,header=None,names=['seq_name']).seq_name.values
            seqs = [seq for seq in seqs if seq in whitelist]
        N = len(seqs)
        if input_params.species_agnostic:
            species_tokens = [0]*N
        else:
            all_species = pd.read_csv(input_params.species_list,names=['species_name']).species_name.sort_values().unique()
            all_species_enc = {x:idx for idx,x in enumerate(all_species)}
            species_tokens = [all_species_enc[x.split(':')[1]] if ':' in x else all_species_enc['Homo_sapiens'] for x in seqs]
        dataset = seq_datasets.FASTADataset(dataset_url, seqs_list=seqs, species_tokens=species_tokens, transform=transform,
                                            max_tokens=input_params.max_tokens, max_overlap_tokens=input_params.max_overlap_tokens,
                                            rank=rank, world_size=world_size, size=N)
    else:
        #for large text datasets
        #{input_params.train_dataset}.idx should have sequence names
        assert os.path.getctime(dataset_url + '.idx') > os.path.getctime(dataset_url), 'Index file is older than dataset file'
        N = int(subprocess.check_output(f'wc -l {dataset_url}.idx',shell=True).decode().split()[0])
        dataset = seq_datasets.TextDataset(dataset_url, transform=transform,
                                            max_tokens=input_params.max_tokens, max_overlap_tokens=input_params.max_overlap_tokens,
                                            rank=rank, world_size=world_size, size=N)
    return N, dataset

if input_params.train_dataset is not None:

    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.max_tokens, total_len = input_params.max_tokens,
                                                  mask_rate = input_params.mlm_probability, split_mask = True)

    N_train, train_dataset = get_dataset(input_params.train_dataset, transform=seq_transform, rank=input_params.rank, world_size=input_params.world_size)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size,
                                  num_workers = 1,  collate_fn = None, shuffle = False) #num_workers MUST be 1

    if input_params.val_dataset is not None:

        seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.max_tokens, total_len = input_params.max_tokens,
                                                      mask_rate = input_params.mlm_probability, split_mask = True)

        N_test, test_dataset = get_dataset(input_params.val_dataset, transform=seq_transform, rank=0, world_size=1)
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size,
                                     num_workers = 1,   collate_fn = None, shuffle = False)


elif input_params.test_dataset is not None:

    if input_params.get_probs or input_params.get_embeddings:

        if input_params.mask_at_test:
            seq_transform = sequence_encoders.RollingMasker(mask_stride = 50, frame = 0)
        else:
            seq_transform = sequence_encoders.PlainOneHot(frame = 0, padding = 'max_length', total_len=input_params.max_tokens)

        batch_size = 1

    else:

        seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.max_tokens, total_len = input_params.max_tokens,
                                                  mask_rate = input_params.mlm_probability, split_mask = True)
        batch_size = input_params.batch_size

    N_test, test_dataset = get_dataset(input_params.test_dataset, transform=seq_transform, rank=0, world_size=1)

    test_dataset.max_overlap_tokens = 0 #no need for overlap when testing

    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size,
                                 num_workers = 1,   collate_fn = None, shuffle = False)

gc.collect()
torch.cuda.empty_cache()

print(f'rank {input_params.rank}:initializing model')

species_encoder = SpecAdd(embed = True, encoder = 'label', d_model = input_params.d_model)

model = DSSResNetEmb(d_input = 5, d_output = 5, d_model = input_params.d_model, n_layers = input_params.n_layers,
                     dropout = input_params.dropout, embed_before = True, species_encoder = species_encoder)

if input_params.checkpoint_dir:

    model.load_state_dict(torch.load(input_params.checkpoint_dir + '/model.pt'))

torch.cuda.set_device(input_params.gpu)

model.cuda()

print(f'rank {input_params.rank}:initializing ddp')

if DDP:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[input_params.gpu])
    model_params = [p for p in model.module.parameters() if p.requires_grad]
else:
    model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(model_params, lr=input_params.max_lr, weight_decay = input_params.weight_decay)

scheduler = None

last_epoch = 0
last_step  = 0 #steps done so far, counted throughout all epochs
last_tokens = 0 #tokens done so far, counted throughout all epochs
last_seqs = 0 #sequences done so far, counted throughout all epochs

scaler = GradScaler(enabled={input_params.mixed_precision})

if input_params.checkpoint_dir:

    if os.path.isfile(input_params.checkpoint_dir + '/optimizer.pt'):
            optimizer.load_state_dict(torch.load(input_params.checkpoint_dir + '/optimizer.pt'))
    if os.path.isfile(input_params.checkpoint_dir + '/scheduler.pt'):
            scheduler.load_state_dict(torch.load(input_params.checkpoint_dir + '/scheduler.pt'))
    if os.path.isfile(input_params.checkpoint_dir + '/scaler.pt'):
            scaler.load_state_dict(torch.load(input_params.checkpoint_dir + '/scaler.pt'))

    if os.path.isfile(input_params.checkpoint_dir + '/meta.json'):
        with open(input_params.checkpoint_dir + '/meta.json') as f:
            meta = json.load(f)
            last_epoch, last_step, last_seqs, last_tokens = meta['epoch'], meta['steps'], meta['sequences'], meta['tokens']

if input_params.rank==0:
    print()
    print(f'Last epoch: {last_epoch}')
    print(f'Steps processed: {last_step:,}')
    print(f'Sequences processed: {last_seqs:,}')
    print(f'Tokens processed: {last_tokens:,}')
    print()

if input_params.train_dataset is not None:
    train_dataloader.dataset.start_seq_idx = last_seqs%N_train #start at a given sequence of the dataset

weights_dir = os.path.join(input_params.output_dir, 'checkpoints') #dir to save model weights at save_at checkpoints

print = misc.LogPrinter(reset=False).print #print elapsed time before the print message

if DDP:
    dist.barrier() #wait for all processes

def chkpt_callback(epoch, tot_steps, tokens_current_run, seqs_current_epoch, model, optimizer, scheduler, train_metrics):

    global last_epoch

    dist.barrier() #wait for all processes

    loss, total_acc, masked_acc = train_metrics

    #dist.all_reduce(tot_steps, op=dist.ReduceOp.SUM)
    dist.all_reduce(seqs_current_epoch, op=dist.ReduceOp.MIN)
    dist.all_reduce(tokens_current_run, op=dist.ReduceOp.SUM) #tokens from all processes in the current training run
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(masked_acc, op=dist.ReduceOp.SUM)

    if input_params.rank!=0:
        return

    tot_seqs = epoch*N_train + seqs_current_epoch - 1 #sequences processed so far, -1 to account for incomplete sequences

    tot_tokens = last_tokens + tokens_current_run #add tokens processed at loaded checkpoint

    chkpt = tot_steps // input_params.steps_per_chkpt

    print(f'checkpoint {chkpt} - train (epoch: {epoch}; sequences: {tot_seqs}; steps: {tot_steps:,}; tokens: {tot_tokens:,}), loss: {loss/input_params.world_size:.4}, total acc: {total_acc/input_params.world_size:.3f}, masked acc: {masked_acc/input_params.world_size:.3f}')

    chkpt = chkpt.item()

    if epoch!=last_epoch:
        print(f'Epoch change between checkpoints {chkpt-1} and {chkpt}')
        #when the epoch changes
        #add this and previous checkpoints s.t. they are not erased
        input_params.save_at.append(max(chkpt-1,0))
        input_params.save_at.append(chkpt)
        last_epoch = epoch.item()

    if (chkpt in input_params.save_at or -1 in input_params.save_at): #save model weights

        checkpoint_dir = misc.save_model_weights(
                        model = model.module if DDP else model,
                        tokenizer = None,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        scaler = scaler,
                        output_dir = weights_dir,
                        checkpoint_name = f'chkpt_{chkpt}',
                        meta = {'epoch':epoch.item(),
                                'sequences':tot_seqs.item(),
                                'steps':tot_steps.item(),
                                'tokens':tot_tokens.item(),
                                'N_train':N_train,
                                'checkpoint':chkpt},
                        save_at = input_params.save_at
                        )

    if  input_params.val_dataset is not None and (chkpt==input_params.tot_chkpt or
                            chkpt%input_params.validate_every==0):

        print(f'CHECKPOINT {chkpt}: Validating...')

        val_metrics, _ =  train_eval.model_eval(model, optimizer, test_dataloader, silent = True)

        loss, total_acc, masked_acc = val_metrics

        print(f'checkpoint {chkpt} - validation, loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}')

        model.train() #model back to train mode

if input_params.train_dataset is not None:

    if input_params.save_at:
        os.makedirs(weights_dir, exist_ok = True)

    print(f'rank {input_params.rank}: start training...')

    with model.join():

        train_eval.model_train(
                model = model,
                optimizer = optimizer,
                dataloader = train_dataloader,
                scheduler = scheduler,
                scaler = scaler,
                last_step = last_step,
                last_epoch = last_epoch,
                grad_accum_itr = input_params.grad_accum_itr,
                steps_per_chkpt = input_params.steps_per_chkpt,
                tot_chkpt = input_params.tot_chkpt,
                chkpt_callback = chkpt_callback,
                mixed_precision = input_params.mixed_precision,
                silent = True,
                )

    print(f'rank {input_params.rank}: training done')

if input_params.rank==0 and input_params.test_dataset is not None:

    print(f'Test/Inference...')

    loss, total_acc, masked_acc =  train_eval.model_eval(model, optimizer, test_dataloader,
                                                          input_params,
                                                          silent = True)


    print(f'test, loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}')
