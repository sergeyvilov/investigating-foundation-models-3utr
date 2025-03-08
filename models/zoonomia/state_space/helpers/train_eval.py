import torch

import numpy as np

from torch import nn

from helpers.metrics import MaskedAccuracy

from helpers.misc import EMA,get_file_info

from torch.nn.functional import log_softmax, softmax

from torch.cuda.amp import autocast

from collections import defaultdict

import os

import pickle

def model_train(model,
                optimizer,
                dataloader,
                scaler, #instance of GradScaler for mixed precision
                steps_per_chkpt, #steps per checkpoint
                tot_chkpt, #total checkpoints for training, aka total epochs
                last_step=0, #total steps processed at loaded checkpoint
                last_epoch=0, #last epoch
                grad_accum_itr=1, #number of batches for gradient accumulation
                scheduler=None,
                silent=False, #display progressbar
                mixed_precision=False, #mixed precision training flag
                chkpt_callback=None #callback function to invoke at each checkpoint
                ):

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    def run_fwd_bwd(inputs,itr_idx):
        '''
        Run forward and backward pass through the network
        '''
        with autocast(enabled=mixed_precision):
            logits, _ = model(inputs["input_ids"],inputs["species_label"])
            loss = criterion(logits, inputs["labels"])
            loss = loss/grad_accum_itr
        if loss.isnan():
            raise ValueError(f'NAN loss in iteration {itr_idx}')
        scaler.scale(loss).backward()
        return logits, loss

    metric = MaskedAccuracy().cuda()

    model.train() #model to train mode

    if not silent:
        #creat progress bar
        from tqdm.notebook import tqdm
        tot_steps = steps_per_chkpt
        pbar = tqdm(total = tot_steps, ncols=700) #progress bar

    loss_EMA = EMA() #exponential moving average loss

    masked_acc, total_acc = 0., 0.,

    tot_steps = last_step #total steps done so far

    n_tokens = 0 #total tokens processed in the current run

    itr_idx = 0 #iterations counter for gradient accumulation

    while True: #infinite loop over dataset
        for inputs in dataloader:

            dataloader.dataset.start_seq_idx = 0 #clear start_seq_idx at the first batch, we don't need to skip previously processed sequences in further epochs

            itr_idx += 1 #increase batch counter

            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['labels'] = inputs['labels'].cuda()
            inputs['labels_unmasked'] = inputs['labels_unmasked'].cuda()
            inputs['species_label'] = inputs['species_label'].cuda()
            inputs['n_tokens'] = inputs['n_tokens'].cuda()
            inputs['seq_idx'] = inputs['seq_idx'].cuda()

            #https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            #https://gist.github.com/mcarilli/bf013d2d2f4b4dd21ade30c9b52d5e2e

            n_tokens += inputs["n_tokens"].sum() #number of meaningful tokens, excluding PAD and CLS

            if itr_idx % grad_accum_itr==0:
                logits, loss = run_fwd_bwd(inputs,itr_idx)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                with model.no_sync():
                    logits, loss = run_fwd_bwd(inputs,itr_idx)
                continue #exit here to allow gradient accumulation across several batches

            if scheduler is not None:
                scheduler.step()

            smoothed_loss = loss_EMA.update(loss.item())

            preds = torch.argmax(logits, dim=1)

            masked_acc += metric(preds, inputs["labels"]).detach() # compute only on masked nucleotides
            total_acc += metric(preds, inputs["labels_unmasked"]).detach()

            tot_steps += 1 # increase step counter every grad_accum_itr iterations

            if not silent:
                pbar.n = tot_steps-last_step
                pbar.set_description(f"acc: {total_acc/(tot_steps-last_step):.3}, masked acc: {masked_acc/(tot_steps-last_step):.3}, loss: {smoothed_loss:.4}")
                pbar.refresh()

            if  tot_steps%steps_per_chkpt==0:
                if chkpt_callback is not None:
                    train_metrics = (torch.tensor(smoothed_loss).cuda(),
                                (total_acc/(tot_steps-last_step)).cuda(),
                                (masked_acc/(tot_steps-last_step)).cuda())
                    chkpt_callback( torch.tensor(last_epoch).cuda(),
                                    torch.tensor(tot_steps).cuda(),
                                    n_tokens.clone(),  #total number of tokens processed so far
                                    inputs['seq_idx'][-1]+1, #total number of sequences processed in the current epoch
                                    model, optimizer, scheduler,
                                    train_metrics)
                loss_EMA = EMA()
                masked_acc, total_acc = 0., 0.
                last_step = tot_steps

            if  tot_steps // steps_per_chkpt == tot_chkpt:
                #all done
                if not silent:
                    del pbar
                return 0

        last_epoch += 1


def model_eval(model, optimizer, dataloader, input_params, silent=False):

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    metric = MaskedAccuracy().cuda()

    model.eval() #model to train mode

    if not silent:
        from tqdm.notebook import tqdm
        tot_steps = len(dataloader.dataset)
        pbar = tqdm(total = tot_steps, ncols=700) #progress bar

    avg_loss, masked_acc, total_acc = 0., 0., 0.

    all_seqs, all_probs, all_embeddings, all_losses = defaultdict(str), defaultdict(list), {}, {}
    seq_names = []
    
    motif_probas = []

    with torch.no_grad():

        for step_idx, inputs in enumerate(dataloader):

            masked_sequence = inputs['input_ids'].cuda()
            targets_masked = inputs['labels'].cuda()
            targets = inputs['labels_unmasked'].cuda()
            species_label = inputs['species_label'].cuda()

            if input_params.get_probs or input_params.get_embeddings:
                #batches are generated by transformation in the dataset,
                #so remove extra batch dimension added by dataloader
                masked_sequence, targets_masked, targets = masked_sequence[0], targets_masked[0], targets[0]
                species_label = species_label.tile((len(masked_sequence),))

            logits, embeddings = model(masked_sequence, species_label)

            loss = criterion(logits, targets_masked)
            
            if loss.isnan():
                raise ValueError(f'NAN loss in steps {step_idx}')

            avg_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
            total_acc += metric(preds, targets).detach()

            seq_name = inputs['seq_name'][0]

            seq_names.append(seq_name)
                    
            if  input_params.get_embeddings:

                # only get embeddings of the masked nucleotide
                sequence_embedding = embeddings["seq_embedding"]
                sequence_embedding = sequence_embedding.transpose(-1,-2)[targets_masked!=-100]
                # shape # B, L, dim  to L,dim, left with only masked nucleotide embeddings
                # average over sequence
                #print(sequence_embedding.shape)
                if input_params.mask_at_test:
                    sequence_embedding = sequence_embedding.mean(dim=0) # if we mask
                else:
                    sequence_embedding = sequence_embedding[0].mean(dim=-1) # no mask

                sequence_embedding = sequence_embedding.detach().cpu().numpy()

                if not seq_name in all_embeddings.keys():
                    all_embeddings[seq_name] = sequence_embedding
                    all_losses[seq_name] = loss.item()

            if  input_params.get_probs:

                logits = torch.permute(logits,(2,0,1)).reshape(-1,5).detach()
                
                targets_masked = targets_masked.T.flatten()
                
                masked_targets = targets_masked[targets_masked!=-100].cpu()
                logits = logits[targets_masked!=-100].cpu()
                
                probs = softmax(logits, dim=1).numpy()

                seq = inputs['seq'][0]
                left_shift = inputs['left_shift'][0]

                all_seqs[seq_name]+=seq[left_shift:]
                all_probs[seq_name].extend(probs[left_shift:])

                if input_params.predict_only_lowercase:
                    assert input_params.central_window is None
                    lower_idx = np.array([idx for idx, c in enumerate(seq) if c.islower()])
                    left = lower_idx.min()
                    right = lower_idx.max()+1
                    all_seqs[seq_name] = all_seqs[seq_name][left:right]
                    all_probs[seq_name] = all_probs[seq_name][left:right]
                        
                if input_params.central_window:
                    assert input_params.predict_only_lowercase is None
                    left = max(len(seq)-input_params.central_window,0)//2
                    right = left + input_params.central_window
                    all_seqs[seq_name] = all_seqs[seq_name][left:right]
                    all_probs[seq_name] = all_probs[seq_name][left:right]
                    
            seq_idx = inputs['seq_idx'][-1].item()
            
            if not silent:
                pbar.n = seq_idx
                pbar.set_description(f"acc: {total_acc/(step_idx+1):.3}, masked acc: {masked_acc/(step_idx+1):.3}, loss: {avg_loss/(step_idx+1):.4}")
                pbar.refresh()
            else:
                print(f'{seq_idx}/{len(dataloader.dataset)}')

    if not silent:
        del pbar

    if input_params.get_probs:
        
        all_probs = [np.array(all_probs[seq_name]) for seq_name in seq_names]
        all_seqs = [all_seqs[seq_name] for seq_name in seq_names]

    if input_params.get_embeddings:
        all_embeddings = [all_embeddings[seq_name] for seq_name in seq_names]
        all_losses = [all_losses[seq_name] for seq_name in seq_names]

    if input_params.get_probs or input_params.get_embeddings:
        
        os.makedirs(input_params.output_dir, exist_ok = True)

        with open(input_params.output_dir + '/predictions.pickle', 'wb') as f:
            pickle.dump({'seq_names':seq_names,
                        'seqs':all_seqs,
                        'embeddings':all_embeddings,
                        'probs':all_probs,
                        'losses':all_losses,
                        'dataset':get_file_info(input_params.test_dataset)},
                        f)

    return avg_loss/(step_idx+1), total_acc/(step_idx+1), masked_acc/(step_idx+1)
