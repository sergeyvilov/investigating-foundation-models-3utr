import torch

import numpy as np

from torch import nn

from helpers.metrics import MaskedAccuracy

from helpers.misc import EMA

from torch.nn.functional import log_softmax
    
def model_train(model, optimizer, dataloader, device, scheduler=None, silent=False):

    metric = MaskedAccuracy().to(device)
    
    model.train() #model to train mode

    if not silent:
        from tqdm.notebook import tqdm
        tot_itr = len(dataloader.dataset)
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    loss_EMA = EMA()
    
    masked_acc, total_acc = 0., 0., 
        
    for itr_idx, inputs in enumerate(dataloader):

        input_ids=inputs["input_ids"].to(device)
        targets_masked = inputs["labels"].to(device)
        #targets = inputs["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            labels=targets_masked,
            attention_mask=inputs["attention_mask"].to(device),
            token_type_ids=inputs["token_type_ids"].to(device)
        )
            
        loss = outputs.loss

        if loss.isnan():
            raise ValueError(f'NAN loss in iteration {itr_idx}')
                
        optimizer.zero_grad()
        
        loss.backward()

        #if max_abs_grad:
        #    torch.nn.utils.clip_grad_value_(model.parameters(), max_abs_grad)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            
        smoothed_loss = loss_EMA.update(loss.item())
            
        preds = torch.argmax(outputs.logits, dim=2)
        
        masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
        total_acc += metric(preds, input_ids).detach()
        
        if not silent:

            pbar.n = inputs['seq_idx'][-1].item()
            pbar.set_description(f"acc: {total_acc/(itr_idx+1):.3}, masked acc: {masked_acc/(itr_idx+1):.3}, loss: {smoothed_loss:.4}")
            pbar.refresh()

         
    if not silent:
        del pbar
        
    print(itr_idx+1)
    
    return smoothed_loss, total_acc/(itr_idx+1), masked_acc/(itr_idx+1) 


def model_eval(model, optimizer, dataloader, device, get_embeddings = False, temperature=None, silent=False):
    
    metric = MaskedAccuracy().to(device)

    model.eval() #model to train mode

    if not silent:
        from tqdm.notebook import tqdm
        tot_itr = len(dataloader.dataset)
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    avg_loss, masked_acc, total_acc = 0., 0., 0.
            
    with torch.no_grad():

        for itr_idx, inputs in enumerate(dataloader):
            
            input_ids=inputs["input_ids"].to(device)
            targets_masked = inputs["labels"].to(device)
            #targets = inputs["targets"].to(device)
    
            outputs = model(
                input_ids=input_ids,
                labels=targets_masked,
                attention_mask=inputs["attention_mask"].to(device),
                token_type_ids=inputs["token_type_ids"].to(device)
            )
            
            loss = outputs.loss

            if loss.isnan():
                raise ValueError(f'NAN loss in iteration {itr_idx}')

            avg_loss += loss.item()
                
            preds = torch.argmax(outputs.logits, dim=2)

            masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
            total_acc += metric(preds, input_ids).detach()
                             
            if not silent:
                
                pbar.n = inputs['seq_idx'][-1].item()
                pbar.set_description(f"acc: {total_acc/(itr_idx+1):.3}, masked acc: {masked_acc/(itr_idx+1):.3}, loss: {avg_loss/(itr_idx+1):.4}")
                pbar.refresh()


    if not silent:
        del pbar
     
    return avg_loss/(itr_idx+1), total_acc/(itr_idx+1), masked_acc/(itr_idx+1)