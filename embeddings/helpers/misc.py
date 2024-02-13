import builtins
import time
import sys

class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def print(*args, **kwargs):
    '''
    Redefine print function for logging
    '''
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #current date and time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()

def center_seq(seq, new_length):

    centered_seq = ['N']*new_length 

    center_pos = len(seq)//2

    left_seq = seq[max(0,center_pos-new_length//2):center_pos] 
    right_seq = seq[center_pos:center_pos+new_length//2] 
    
    centered_seq[new_length//2:new_length//2+len(right_seq)] =  right_seq
    centered_seq[new_length//2-len(left_seq):new_length//2] = left_seq

    return ''.join(centered_seq)
