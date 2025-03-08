import torch

import numpy as np

from torch import nn

from helpers.metrics import MaskedAccuracy

from helpers.misc import EMA

from torch.nn.functional import log_softmax

from torch.cuda.amp import autocast

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

    def run_fwd_bwd(inputs,itr_idx):
        '''
        Run forward and backward pass through the network
        '''
        with autocast(enabled=mixed_precision):
            outputs = model(
                    input_ids=inputs["input_ids"],
                    labels=inputs["labels"],
                    )
            loss = outputs.loss/grad_accum_itr
        if loss.isnan():
            raise ValueError(f'NAN loss in iteration {itr_idx}')
        scaler.scale(loss).backward()
        return outputs.logits, loss

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

            inputs = {k:v.cuda() for k,v in inputs.items()}

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

            preds = torch.argmax(logits, dim=2)

            masked_acc += metric(preds, inputs["labels"]).detach() # compute only on masked nucleotides
            total_acc += metric(preds, inputs["input_ids"]).detach()

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


def model_eval(model, optimizer, dataloader, device, get_embeddings = False, silent=False):

    metric = MaskedAccuracy().cuda()

    model.eval() #model to train mode

    if not silent:
        from tqdm.notebook import tqdm
        tot_steps = len(dataloader.dataset)
        pbar = tqdm(total = tot_steps, ncols=700) #progress bar

    avg_loss, masked_acc, total_acc = 0., 0., 0.

    with torch.no_grad():

        for step_idx, inputs in enumerate(dataloader):

            input_ids=inputs["input_ids"].cuda()
            targets_masked = inputs["labels"].cuda()
            #targets = inputs["targets"].cuda()

            outputs = model(
                input_ids=input_ids,
                labels=targets_masked,
            )

            loss = outputs.loss

            if loss.isnan():
                raise ValueError(f'NAN loss in steps {step_idx}')

            avg_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=2)

            masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
            total_acc += metric(preds, input_ids).detach()

            if not silent:
                pbar.n = inputs['seq_idx'][-1].item()
                pbar.set_description(f"acc: {total_acc/(step_idx+1):.3}, masked acc: {masked_acc/(step_idx+1):.3}, loss: {avg_loss/(step_idx+1):.4}")
                pbar.refresh()


    if not silent:
        del pbar

    return avg_loss/(step_idx+1), total_acc/(step_idx+1), masked_acc/(step_idx+1)
