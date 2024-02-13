import pickle
import time
import builtins
import sys
import torch
import os
import json

class EMA():
    '''
    Exponential moving average
    '''
    def __init__(self, beta = 0.98):
        
        self.beta = beta
        self.itr_idx = 0
        self.average_value = 0
        
    def update(self, value):
        self.itr_idx += 1
        self.average_value = self.beta * self.average_value + (1-self.beta)*value
        smoothed_average = self.average_value / (1 - self.beta**(self.itr_idx))
        return smoothed_average
    
class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def list2range(v):
    r = []
    for num in v:
        if not ':' in num:
            r.append(int(num))
        else:
            k = [int(x) for x in num.split(':')]
            if len(k)==2:
                r.extend(list(range(k[0],k[1]+1)))
            else:
                r.extend(list(range(k[0],k[1]+1,k[2])))
    return r



def print_log(*args, **kwargs):
    '''
    Redefine print function for logging
    '''
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #current date and time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()

def print_pass(*args, **kwargs):
    pass
            
def save_model_weights(model, optimizer, scheduler, output_dir, epoch):
    '''
    Save model and optimizer weights
    '''
    config_save_base = os.path.join(output_dir, f'epoch_{epoch}')

    print(f'SAVING MODEL, CHECKPOINT DIR: {config_save_base}\n')

    model.save_pretrained(config_save_base)
    
    #torch.save(model.state_dict(), config_save_base+'_model') #save model weights

    torch.save(optimizer.state_dict(), config_save_base+'/optimizer.pt') #save optimizer weights

    torch.save(scheduler.state_dict(), config_save_base+'/scheduler.pt') #save optimizer weights

    
