#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from datasets import Dataset

import math
import itertools
from collections.abc import Mapping
import numpy as np
import pandas as pd
import sys
import pickle
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "fasta file name", type = str, required = True)

parser.add_argument("--checkpoint_dir", help = "checkpoint dir", type = str, required = True)

parser.add_argument("--output_dir", help = "output dir name", type = str, required = True)

#parser.add_argument("--trimseq", help = "trim sequence to match input length (510)", action='store_true', default = False, required = False)

parser.add_argument("--N_folds", help = "number of folds to split sequences", type = int, default = None, required = False)

parser.add_argument("--fold", help = "current fold", type = int, default = None, required = False)

parser.add_argument("--reverse_seq_neg_strand", help = "reverse-complement sequences on the negative strand before inference", action='store_true', default = False, required = False)

parser.add_argument("--strand_bed", help = "bed file with sequence strand information, used with reverse_seq_neg_strand ", type = str, required = False)

parser.add_argument("--central_window", help = "perform inference only for central_window nucleotides around the sequence center, assign nan probability to all other positions", type = int, default=False, required = False)

parser.add_argument("--predict_only_lowercase", help = "keep predictions only for the lowercased positions", action='store_true', default = False, required = False)

parser.add_argument("--crop_lowercase", help = "crop the sequences s.t. they are centered on the lowecase part", action='store_true', default = False, required = False)

parser.add_argument("--crop_center", help = "crop the sequences s.t. they are centered", action='store_true', default = False, required = False)

parser.add_argument("--whitelist", help = "include sequences only from this list", type = str, default=None, required = False)

input_params = parser.parse_args()

assert not (input_params.predict_only_lowercase and input_params.central_window)
assert not (input_params.crop_lowercase and input_params.crop_center)

print(input_params)

print(f'Reverse sequences on negative strand before inference: {input_params.reverse_seq_neg_strand}')

tokenizer = AutoTokenizer.from_pretrained(input_params.checkpoint_dir)
model = AutoModelForMaskedLM.from_pretrained(input_params.checkpoint_dir)


# # Utility Functions

# ## Tokenization

# In[2]:


def chunkstring(string, length):
    # chunks a string into segments of length
    return (string[0+i:length+i] for i in range(0, len(string), length))

def kmers(seq, k=6):
    # splits a sequence into non-overlappnig k-mers
    return [seq[i:i + k] for i in range(0, len(seq), k) if i + k <= len(seq)]

def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]

def tok_func(x): return tokenizer(" ".join(kmers_stride1(x["seq_chunked"])))

def one_hot_encode(gts, dim=5):
    result = []
    nuc_dict = {"A":0,"C":1,"G":2,"T":3}
    for nt in gts:
        vec = np.zeros(dim)
        vec[nuc_dict[nt]] = 1
        result.append(vec)
    return np.stack(result, axis=0)

nuc_dict = {"A":0,"C":1,"G":2,"T":3}
def class_label_gts(gts):
    return np.array([nuc_dict[x] for x in gts])


# In[3]:


#from transformers import  DataCollatorForLanguageModeling
#data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability = 0.15)
torch.manual_seed(0)

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class DataCollatorForLanguageModelingSpan():
    def __init__(self, tokenizer, mlm, mlm_probability, span_length):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.span_length =span_length
        self.mlm_probability= mlm_probability
        self.pad_to_multiple_of = span_length

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        import torch

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability*0.2)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool().numpy()
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * self.span_length, mode = 'same' ),axis = 1, arr = masked_indices).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        m_save = masked_indices.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability*0.8)
        probability_matrix.masked_fill_(masked_indices, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool().numpy()
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * self.span_length, mode = 'same' ),axis = 1, arr = masked_indices).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        m_final = masked_indices + m_save
        labels[~m_final] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        #indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool()
        #print (indices_replaced)
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        #print (masked_indices)

        # 10% of the time, we replace masked input tokens with random word
        #indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        #random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        #inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# ## Prediction

# In[4]:


def predict_on_batch(tokenized_data, dataset, seq_idx, crop_mask_left=None, crop_mask_right=None):
    model_input_unaltered = tokenized_data['input_ids'].clone()
    label = dataset.iloc[seq_idx]['seq_chunked']
    L = len(label)
    if L < 6:
        return torch.ones(L,L,5)*0.25, None
    else:
        diag_matrix = torch.eye(tokenized_data['input_ids'].shape[1]).numpy()
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * 6, mode = 'same' ),axis = 1, arr = diag_matrix).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        masked_indices = masked_indices[3:L-5-2]
        if crop_mask_left and crop_mask_right:
                masked_indices = masked_indices[crop_mask_left:crop_mask_right]
        res = tokenized_data['input_ids'].expand(masked_indices.shape[0],-1).clone()
        targets_masked = res.clone().to(device)
        res[masked_indices] = 4
        targets_masked[res!=4] = -100
        #print (res[0], res.shape)
        res = res.to(device)
        with torch.no_grad():
            model_outs = model(res,labels=targets_masked)
            fin_calculation = torch.softmax(model_outs['logits'], dim=2).detach().cpu()
        return fin_calculation, model_outs['loss']

# ## Translating predictions

# In[5]:


def extract_prbs_from_pred(prediction, pred_pos, token_pos, label_pos, label):
    # pred_pos = "kmer" position in tokenized sequence (incl. special tokens)
    # token_pos = position of nucleotide in kmer
    # label_pos = position of actual nucleotide in sequence
    model_pred = prediction
    prbs = [torch.sum(model_pred[pred_pos,tokendict_list[token_pos][nuc]]) for nuc in ["A","C","G","T"]]
    gt = label[label_pos] # 6-CLS, zerobased
    res = torch.tensor(prbs+[0.0])
    return res, gt


# # Prepare inputs
def reverse_complement(seq):
    '''
    Take sequence reverse complement
    '''
    compl_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    compl_seq = ''.join([compl_dict.get(x,x) for x in seq])
    rev_seq = compl_seq[::-1]
    return rev_seq

# ## Prepare dataframe

# In[6]:


# In[39]:


dataset = defaultdict(str)

with open(input_params.fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            seq_name = line[1:].rstrip()
        else:
            dataset[seq_name] += line.rstrip()#.upper()

dataset = pd.DataFrame(list(dataset.items()), columns=['seq_name','seq'])

if input_params.whitelist:
    whitelist = pd.read_csv(input_params.whitelist,header=None,names=['seq_name']).seq_name.values
    dataset = dataset[dataset.seq_name.isin(whitelist)]
    print(len(dataset))

if input_params.N_folds:
    folds = np.arange(input_params.N_folds).repeat(len(dataset)//input_params.N_folds+1)[:len(dataset)]

    dataset = dataset.loc[folds==input_params.fold]

    print(f'Fold {input_params.fold}: {len(dataset)} sequences')

MAX_LEN = 508

if input_params.crop_center or input_params.crop_lowercase:
    if input_params.crop_center:
        dataset['center_pos'] = dataset.seq.apply(len)//2
    if input_params.crop_lowercase:
        dataset['center_pos'] = dataset.seq.apply(lambda x:np.median([idx for idx,c in enumerate(x) if c.islower()]))
    dataset['left_shift'] = dataset.center_pos.apply(lambda x:max(x-MAX_LEN//2,0)).astype(int)
    dataset['seq'] = dataset.apply(lambda x : x.seq[x.left_shift:x.left_shift+MAX_LEN],axis=1)
else: 
    dataset['left_shift'] = 0

if input_params.central_window:
    dataset['pos_left'] = dataset.seq.apply(lambda x : max(len(x)//2-input_params.central_window//2,0))
    dataset['pos_right'] = dataset.apply(lambda x : min(x.pos_left+input_params.central_window,len(x.seq)),axis=1)
elif input_params.predict_only_lowercase:
    dataset['pos_left'] = dataset.seq.apply(lambda x : np.min([idx for idx,c in enumerate(x) if c.islower()]))
    dataset['pos_right'] = dataset.seq.apply(lambda x : np.max([idx for idx,c in enumerate(x) if c.islower()]))+1
else:
    dataset['pos_left'] = None
    dataset['pos_right'] = None  
    
dataset['original_seq'] = dataset['seq'].copy()

dataset.seq = dataset.seq.apply(lambda x:x.upper())

if input_params.reverse_seq_neg_strand:
    strand_info = pd.read_csv(input_params.strand_bed, sep='\t', header = None, names=['seq_name','strand'], usecols=[3,5]).set_index('seq_name').squeeze()

    dataset.seq = dataset.apply(lambda x: reverse_complement(x.seq) if strand_info.loc[x.seq_name]=='-' else x.seq, axis=1) #undo reverse complement

dataset['seq_chunked'] = dataset['seq'].apply(lambda x : list(chunkstring(x, MAX_LEN))) 


# In[42]:


# In[43]:


dataset = dataset.explode('seq_chunked')


# In[44]:


ds = Dataset.from_pandas(dataset[['seq_chunked']])

tok_ds = ds.map(tok_func, batched=False,  num_proc=2)

rem_tok_ds = tok_ds.remove_columns('seq_chunked')

data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=False, mlm_probability = 0.025, span_length =6)
data_loader = torch.utils.data.DataLoader(rem_tok_ds, batch_size=1, collate_fn=data_collator, shuffle = False)


# ## Prepare model

# In[45]:

print('Initializing CUDA device')

device = torch.device("cuda")
model.to(device)
print ("Done.")


# In[46]:

computed = []

model.eval()
model.to(device)

# ## Prepare tokendict

# In[14]:


tokendict_list = [{"A": [], "G": [], "T": [],"C": []} for x in range(6)]

for tpl in itertools.product("ACGT",repeat=6):
    encoding = tokenizer.encode("".join(tpl))
    for idx, nuc in enumerate(tpl):
        tokendict_list[idx][nuc].append(encoding[1])


# # Run Inference

# In[ ]:

print('Running inference')

k = 6
predicted_prbs,gts,is_correct,loss = [],[],[],[]
#print (dataset.iloc[0]['seq_chunked'])

for no_of_index, tokenized_data in enumerate(data_loader):
    #if no_of_index < 1340:
    #    continue
    label = dataset.iloc[no_of_index]['seq_chunked']
    pos_left = dataset.iloc[no_of_index]['pos_left']
    pos_right = dataset.iloc[no_of_index]['pos_right']

    L = len(label)

    if L < 11: 
        #print (no_of_index)
        for i in range(L):
            predicted_prbs.append(torch.tensor([np.nan,np.nan,np.nan,np.nan,np.nan]))
            gts.append(label[i])
        continue
        
    if (input_params.central_window or input_params.predict_only_lowercase) and pos_left>5 and pos_right<L-5:

        token_idx_left = pos_left-5 #first token, where the leftmost nucleotide within central_window appears
        token_idx_right = token_idx_left + pos_right-pos_left+6-1
    
        predictions,seq_loss = predict_on_batch(tokenized_data, dataset, no_of_index,
                                               crop_mask_left=token_idx_left,crop_mask_right=token_idx_right)
    
        for pos in range(pos_left):
            predicted_prbs.append(torch.tensor([np.nan,np.nan,np.nan,np.nan,np.nan]))
            gts.append(label[pos])
            
        for pos in range(pos_left,pos_right):
                model_pred = predictions[pos-pos_left]
                res,gt = extract_prbs_from_pred(prediction=model_pred,
                                                pred_pos=pos-2, # for i-th nt, we look at (i-2)th 6-mer
                                                token_pos=3, # look at 4th nt in 6-mer
                                                label_pos=pos,
                                                label=label)    
                predicted_prbs.append(res)
                gts.append(gt)
                is_correct.append(res.argmax() == nuc_dict.get(gt,4))
    
        for pos in range(pos_right,L):
            predicted_prbs.append(torch.tensor([np.nan,np.nan,np.nan,np.nan,np.nan]))
            gts.append(label[pos])

        loss.append(np.nan)
        
    else:
        
        model_input_unaltered = tokenized_data['input_ids'].clone()
        tokenized_data['labels'][tokenized_data['labels']==-100] = 0
        inputs = model_input_unaltered.clone()
    
    
        # First 5 nucleotides we infer from the first 6-mer
        inputs[:, 1:7] = 4 # we mask the first 6 6-mers
        inputs = inputs.to(device)
    
        targets_masked = model_input_unaltered.clone().to(device)
        targets_masked[inputs!=4] = -100
    
        with torch.no_grad():
            model_outs = model(inputs,labels=targets_masked)
    
        model_pred = torch.softmax(model_outs['logits'], dim=2)
        loss.append(model_outs['loss'].item())
    
        for i in range(5):
            res,gt = extract_prbs_from_pred(prediction=model_pred[0],
                                            pred_pos=1, # first 6-mer (after CLS)
                                            token_pos=i, # we go thorugh first 6-mer
                                            label_pos=i,
                                            label=label)
            predicted_prbs.append(res)
            gts.append(gt)
            is_correct.append(res.argmax() == nuc_dict.get(gt,4))
    
    
    
        # we do a batched predict to process the rest of the sequence
        predictions,seq_loss = predict_on_batch(tokenized_data, dataset, no_of_index)
        if seq_loss is not None:
            loss.append(seq_loss.item())
    
        # For the 6th nt up to the last 5
        # we extract probabilities similar to how the model was trained
        # hiding the 4th nt of the 3rd masked 6-mer of a span of 6 masked 6-mers
        # note that CLS makes the tokenized seq one-based
        pos = 5 # position in sequence
        for pos in range(5, L-5):
            model_pred = predictions[pos-5]
            res,gt = extract_prbs_from_pred(prediction=model_pred,
                                            pred_pos=pos-2, # for i-th nt, we look at (i-2)th 6-mer
                                            token_pos=3, # look at 4th nt in 6-mer
                                            label_pos=pos,
                                            label=label)
            predicted_prbs.append(res)
            gts.append(gt)
            is_correct.append(res.argmax() == nuc_dict.get(gt,4))
    
        # Infer the last 5 nt from the last 6-mer
        for i in range(5):
            model_pred = predictions[pos-5]
            res,gt = extract_prbs_from_pred(prediction=model_pred,
                                    pred_pos=pos+1, # len - 5 + 1 = last 6-mer (1-based)
                                    token_pos=i+1, # we go through last 5 of last 6-mer
                                    label_pos=pos+i,
                                    label=label)
            predicted_prbs.append(res)
            gts.append(gt)
            is_correct.append(res.argmax() == nuc_dict.get(gt,4))

    assert(len(gts) == torch.stack(predicted_prbs).shape[0]), "{} iter, expected len:{} vs actual len:{}".format(no_of_index,
                                                                                   len(gts),
                                                                     torch.stack(predicted_prbs).shape[0])
    print(f'chunks:{no_of_index+1}, acc:{np.mean(is_correct):.3f}, loss:{np.mean(loss):.3f}')


# In[50]:

print('Saving predictions')

predicted_prbs = np.array(predicted_prbs)[:,:4]

dataset = dataset[['seq_name','original_seq', 'left_shift']].drop_duplicates()
dataset['seq_len'] = dataset.original_seq.apply(len)

all_preds = []

s = 0

for seq_name, original_seq, left_shift, seq_len in dataset.values.tolist():
    seq_probas = predicted_prbs[s:s+seq_len,:]
    s += seq_len
    if  input_params.reverse_seq_neg_strand and strand_info[seq_name]=='-':
        seq_probas = seq_probas[::-1,[3,2,1,0]] #reverse complement probabilities s.t. probas match original_seq
    all_preds.append((seq_name,original_seq, left_shift, seq_probas))

if input_params.N_folds:
    output_name = input_params.output_dir + f'/predictions_{input_params.fold}.pickle'
else:
    output_name = input_params.output_dir + f'/predictions.pickle'

with open(output_name, "wb") as f:
    seq_names, seqs, left_shift, probs = zip(*all_preds)
    pickle.dump({'seq_names':seq_names, 'seqs':seqs, 'left_shift':left_shift, 'probs':probs},f)
