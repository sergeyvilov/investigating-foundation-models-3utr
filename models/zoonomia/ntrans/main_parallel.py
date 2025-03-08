import os
import subprocess
import gc
import pickle
import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from tqdm import tqdm

import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_inverse_sqrt_schedule

import helpers.datasets as seq_datasets
import helpers.misc as misc                #miscellaneous functions
import helpers.train_eval as train_eval    #train and evaluation

from ntrans.esm_config import EsmConfig
from ntrans.modeling_esm import EsmForMaskedLM

import torch.distributed as dist

from torch.cuda.amp import GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--train_dataset", help = "txt/FASTA dataset, one sequence per line", type = str, required = False)
parser.add_argument("--val_dataset", help = "txt/FASTA dataset, one sequence per line", type = str, required = False)
parser.add_argument("--test_dataset", help = "txt/FASTA dataset, one sequence per line", type = str, required = False)
parser.add_argument("--species_list", help = "list of all species for the species-aware model", type = str, required = False)
parser.add_argument("--model_name", help = "nucleotide transformer model name, e.g. v2-50m", type = str, required = False)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--checkpoint_dir", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--validate_every", help = "validate every N checkpoints", type = int,  default = 0, required = False)
parser.add_argument("--species_agnostic", help = "use a species agnostic version", action='store_true', default = False, required = False)
parser.add_argument("--shift_augm", help = "data augmentation:shift input sequence by up to 50nt", action='store_true', default = False, required = False)
parser.add_argument("--mixed_precision", help = "mixed precision training", action='store_true', default = False, required = False)
parser.add_argument("--steps_per_chkpt", help = "make this number of steps before the checkpoint", type = int, default = 1000, required = False)
parser.add_argument("--grad_accum_itr", help = "number of iterations for gradient accumulation", type = int, default = 1, required = False)
parser.add_argument("--max_overlap_tokens", help = "maximal tokenized chunk overlap", type = int, default = 10, required = False)
parser.add_argument("--max_tokens", help = "maximal chunk length in tokens", type = int, default = 1024, required = False)
parser.add_argument("--step_size_up", help = "number of steps for lr ascending", type = int, default = 16000, required = False)
parser.add_argument("--step_size_down", help = "number of steps for lr descending", type = int, default = None, required = False)
parser.add_argument("--mlm_probability", help = "masking probability", type = float, default = 0.15, required = False)
parser.add_argument("--tot_chkpt", help = "train for this number of checkpoints", type = int, default = 10000, required = False)
parser.add_argument("--batch_size", help = "batch size per single GPU", type = int, default = 32, required = False)
parser.add_argument("--max_lr", help = "learning rate", type = float, default = 1e-4, required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0, required = False)
parser.add_argument("--save_at", help = "checkpoints to save model/optimizer weights, 1-based", nargs='+', type = str, default = ['1:30'], required = False)
parser.add_argument("--seed", help = "seed for reproducibility", type = int, default = None, required = False)

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
    torch.use_deterministic_algorithms(True)

assert "WORLD_SIZE" in os.environ
assert 'SLURM_PROCID' in os.environ # for slurm scheduler

input_params.world_size = int(os.environ["WORLD_SIZE"])
ngpus_per_node = torch.cuda.device_count()
input_params.rank = int(os.environ['SLURM_PROCID'])
input_params.gpu = input_params.rank % torch.cuda.device_count()
input_params.effective_batch_size = input_params.world_size*input_params.batch_size*input_params.grad_accum_itr

print(f'rank:{input_params.rank}, tot gpus:{ngpus_per_node}, gpu:{input_params.gpu}')

dist.init_process_group(backend='nccl', world_size=input_params.world_size, rank=input_params.rank,)

if input_params.rank==0:

    for param_name in ['output_dir', '\\',
            'model_name', '\\',
            'train_dataset','val_dataset','test_dataset','\\',
            'species_list', '\\',
            'checkpoint_dir', '\\',
            'species_agnostic', '\\',
            'seed', '\\',
            'mixed_precision', '\\',
            'steps_per_chkpt','tot_chkpt', 'save_at', '\\',
            'validate_every', '\\',
            'shift_augm', '\\',
            'max_tokens', 'max_overlap_tokens', 'mlm_probability', '\\',
            'world_size','batch_size', 'grad_accum_itr', 'effective_batch_size', '\\',
            'max_lr', 'weight_decay','step_size_up', 'step_size_down', '\\',
            ]:

                if param_name == '\\':
                    print()
                else:
                    print(f'{param_name.upper()}: {input_params[param_name]}')

input_params.save_at = misc.list2range(input_params.save_at)

tokenizer = AutoTokenizer.from_pretrained('3utr_tokenizer',model_max_length=input_params.max_tok_len)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=input_params.mlm_probability)

def get_dataset(dataset_url, transform={}, rank=input_params.rank, world_size=input_params.world_size):
    if dataset_url.endswith('.fa'):
        #for FASTA datasets
        #datasets should be indexed with samtools
        seqs = pd.read_csv(dataset_url + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name']).seq_name.values
        N = len(seqs)
        if input_params.species_agnostic:
            species_tokens = [0]*N
        else:
            all_species = pd.read_csv(input_params.species_list,names=['species_name']).species_name.sort_values().unique()
            all_species_enc = {x:idx for idx,x in enumerate(all_species)}
            species_tokens = [all_species_enc[x.split(':')[1]] if ':' in x else all_species_enc['Homo_sapiens'] for x in seqs]
        dataset = seq_datasets.FASTADataset(dataset_url, seqs_list=seqs, species_tokens=species_tokens, transform=transform,tokenizer=tokenizer,
                                            max_tokens=input_params.max_tokens, max_overlap_tokens=input_params.max_overlap_tokens,
                                            rank=rank, world_size=world_size, size=N)
    else:
        #for large text datasets
        #{input_params.train_dataset}.idx should have sequence names
        N = int(subprocess.check_output(f'wc -l {dataset_url}.idx',shell=True).decode().split()[0])
        dataset = seq_datasets.TextDataset(dataset_url, transform=transform,tokenizer=tokenizer,
                                            max_tokens=input_params.max_tokens, max_overlap_tokens=input_params.max_overlap_tokens,
                                            rank=rank, world_size=world_size, size=N)
    return N, dataset

if input_params.train_dataset is not None:

    N_train, train_dataset = get_dataset(input_params.train_dataset, transform={'shift_augm':input_params.shift_augm},rank=input_params.rank, world_size=input_params.world_size)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size,
                                  num_workers = 1,  collate_fn = collate_fn, shuffle = False) #num_workers MUST be 1

    if input_params.val_dataset is not None:

        N_test, test_dataset = get_dataset(input_params.val_dataset, rank=0, world_size=1)
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size,
                                     num_workers = 1,   collate_fn = collate_fn, shuffle = False)

gc.collect()
torch.cuda.empty_cache()

print(f'rank {input_params.rank}:initializing model')

if input_params.checkpoint_dir:

    model = EsmForMaskedLM.from_pretrained(input_params.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(input_params.checkpoint_dir)

else:

    config = EsmConfig.from_pretrained(f'ntrans/config_{input_params.model_name}.json')

    config.vocab_size = len(tokenizer)

    model = EsmForMaskedLM(config)

    model.resize_token_embeddings(len(tokenizer))

torch.cuda.set_device(input_params.gpu)

model.cuda()

print(f'rank {input_params.rank}:initializing ddp')

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[input_params.gpu],
                                                gradient_as_bucket_view=True, find_unused_parameters=True)

model_params = [p for p in model.module.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, betas=(0.9,0.999), eps=1e-8,
                             lr = input_params.max_lr,
                             weight_decay = input_params.weight_decay)

if input_params.step_size_down is not None:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0, max_lr = input_params.max_lr,
        step_size_up = input_params.step_size_up, step_size_down = input_params.step_size_down, cycle_momentum=False)
else:
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=input_params.step_size_up,)

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

train_dataloader.dataset.start_seq_idx = last_seqs%N_train #start at a given sequence of the dataset

weights_dir = os.path.join(input_params.output_dir, 'checkpoints') #dir to save model weights at save_at checkpoints

if input_params.save_at:
    os.makedirs(weights_dir, exist_ok = True)

print = misc.LogPrinter(reset=False).print #print elapsed time before the print message

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
                        model = model.module,
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

        _ = os.system('cp ./ntrans/*.py ' + checkpoint_dir)
        _ = os.system('cp ./3utr_tokenizer/* ' + checkpoint_dir)

    if  input_params.val_dataset is not None and (chkpt==input_params.tot_chkpt or
                            chkpt%input_params.validate_every==0):

        print(f'CHECKPOINT {chkpt}: Validating...')

        val_metrics =  train_eval.model_eval(model, optimizer, test_dataloader, silent = True)

        loss, total_acc, masked_acc = val_metrics

        print(f'checkpoint {chkpt} - validation, loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}')

        model.train() #model back to train mode

if not input_params.test:

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

print(f'rank {input_params.rank}: all done')
