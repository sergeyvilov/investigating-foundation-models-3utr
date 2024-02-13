import torch

import numpy as np

from torch import nn

from tqdm import tqdm

from helpers.metrics import MaskedAccuracy

from helpers.misc import EMA

from torch.nn.functional import log_softmax
    
def model_train(model, optimizer, dataloader, device, scheduler=None, silent=False):

    metric = MaskedAccuracy().to(device)
    
    model.train() #model to train mode

    if not silent:
        tot_itr = int(np.ceil(len(dataloader.dataset)/dataloader.batch_size)) #total train iterations
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

            pbar.update(1)
            pbar.set_description(f"acc: {total_acc/(itr_idx+1):.2}, masked acc: {masked_acc/(itr_idx+1):.2}, loss: {smoothed_loss:.4}")
         
    if not silent:
        del pbar
        
    print(itr_idx+1)
    
    return smoothed_loss, total_acc/(itr_idx+1), masked_acc/(itr_idx+1) 


def model_eval(model, optimizer, dataloader, device, get_embeddings = False, temperature=None, silent=False):
    
    metric = MaskedAccuracy().to(device)

    model.eval() #model to train mode

    if not silent:
        tot_itr = int(np.ceil(len(dataloader.dataset)/dataloader.batch_size)) #total train iterations
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

            avg_loss += loss.item()
                
            preds = torch.argmax(outputs.logits, dim=2)

            masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
            total_acc += metric(preds, input_ids).detach()
                             
            if not silent:
                
                pbar.update(1)
                pbar.set_description(f"acc: {total_acc/(itr_idx+1):.2}, masked acc: {masked_acc/(itr_idx+1):.2}, loss: {avg_loss/(itr_idx+1):.4}")

    if not silent:
        del pbar
     
    return avg_loss/(itr_idx+1), total_acc/(itr_idx+1), masked_acc/(itr_idx+1)