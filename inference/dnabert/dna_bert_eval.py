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

fasta = sys.argv[1]
model_dir = sys.argv[2] 
output_dir = sys.argv[3]
fold = int(sys.argv[4])

if not '3utr' in model_dir:
    #dna model: model trained on sequences that weren't reverse complemented for genes on negative strand
    #all default DNABERT models
    dna_model = True
else:
    dna_model = False

print(f'DNA model:{dna_model}')

N_folds = 10

data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'


tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMaskedLM.from_pretrained(model_dir)

print(f'Writing to: {output_dir}')

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


def predict_on_batch(tokenized_data, dataset, seq_idx):
    model_input_unaltered = tokenized_data['input_ids'].clone()
    label = dataset.iloc[seq_idx]['seq_chunked']
    label_len = len(label)
    if label_len < 6:
        return torch.zeros(label_len,label_len,5), None
    else:
        diag_matrix = torch.eye(tokenized_data['input_ids'].shape[1]).numpy()
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * 6, mode = 'same' ),axis = 1, arr = diag_matrix).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        masked_indices = masked_indices[3:label_len-5-2]
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

with open(fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            seq_name = line[1:].rstrip()
        else:
            dataset[seq_name] += line.rstrip().upper()
            
dataset = pd.DataFrame(list(dataset.items()), columns=['seq_name','seq'])

folds = np.arange(N_folds).repeat(len(dataset)//N_folds+1)[:len(dataset)] 

dataset = dataset.loc[folds==fold]

strand_info = pd.read_csv(data_dir + 'UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed', sep='\t', header = None, names=['seq_name','strand'], usecols=[3,5]).set_index('seq_name').squeeze()

dataset['original_seq'] = dataset['seq'] 

dataset['seq'] = dataset.apply(lambda x: reverse_complement(x.seq) if strand_info.loc[x.seq_name]=='-' and dna_model else x.seq, axis=1) #undo reverse complement

dataset['seq_chunked'] = dataset['seq'].apply(lambda x : list(chunkstring(x, 510))) #chunk string in segments of 300


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
    label_len = len(label)
    #print(no_of_index, label_len)
    
    # Edge case: for a sequence less then 11 nt
    # we cannot even feed 6 mask tokens
    # so we might as well predict random
    if label_len < 11: 
        #print (no_of_index)
        for i in range(label_len):
            predicted_prbs.append(torch.tensor([0.25,0.25,0.25,0.25,0.0]))
            gts.append(label[i])
            is_correct.append(res.argmax().item() == nuc_dict.get(gt,4))
        continue

        
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
    for pos in range(5, label_len-5):
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
    #XABCDEFGHIJKL -> XABCDE [ABCDEF BCDEFG CDEFGH DEFGHI EFGHIJ FGHIJK] GHIJKL


# In[50]:

print('Saving predictions')

predicted_prbs = np.array(predicted_prbs)[:,:4]

dataset = dataset[['seq_name','original_seq']].drop_duplicates()
dataset['seq_len'] = dataset.original_seq.apply(len)

all_preds = []

s = 0

for seq_name, original_seq, seq_len in dataset.values.tolist():
    seq_probas = predicted_prbs[s:s+seq_len,:]
    s += seq_len
    if strand_info[seq_name]=='-' and dna_model:
        seq_probas = seq_probas[::-1,[3,2,1,0]] #reverse complement probabilities s.t. probas match original_seq
    all_preds.append((seq_name,original_seq, seq_probas))

with open(output_dir + f"/predictions_{fold}.pickle", "wb") as f:
    seq_names, seqs, probs = zip(*all_preds)
    pickle.dump({'seq_names':seq_names, 'seqs':seqs, 'probs':probs},f)

