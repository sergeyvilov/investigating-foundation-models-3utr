#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import Trainer

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


tokenizer = AutoTokenizer.from_pretrained("/s/project/mll/sergey/effect_prediction/MLM/dnabert/default/6-new-12w-0/")

model = AutoModelForMaskedLM.from_pretrained("/s/project/mll/sergey/effect_prediction/MLM/dnabert/default/6-new-12w-0/")


import torch 
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

from datasets import Dataset

import math
import itertools
from collections.abc import Mapping
import numpy as np
import pandas as pd
import sys



output_dir = sys.argv[1]#'/s/project/mll/sergey/effect_prediction/MLM/baseline/dnabert/default/predictions/'

test_dataset = sys.argv[2]#'/s/project/mll/sergey/effect_prediction/MLM/motif_predictions/split_75_25/test.csv'

dataset_start = int(sys.argv[3])
dataset_len = int(sys.argv[4])
output_logits = int(sys.argv[5])

print(f'Running inference for sequences {dataset_start}-{dataset_start+dataset_len}')

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

def class_label_gts(gts):
    nuc_dict = {"A":0,"C":1,"G":2,"T":3}
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
        return torch.zeros(label_len,label_len,5)
    else:
        diag_matrix = torch.eye(tokenized_data['input_ids'].shape[1]).numpy()
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * 6, mode = 'same' ),axis = 1, arr = diag_matrix).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        masked_indices = masked_indices[3:label_len-5-2]
        res = tokenized_data['input_ids'].expand(masked_indices.shape[0],-1).clone()
        res[masked_indices] = 4
        #print (res[0], res.shape)
        res = res.to(device)
        with torch.no_grad():
            if output_logits:
                fin_calculation = model(res)['logits'].detach().cpu()
            else:
                fin_calculation = torch.softmax(model(res)['logits'], dim=2).detach().cpu()   
        return fin_calculation


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

# ## Prepare dataframe

# In[6]:


# In[39]:


dataset = pd.read_csv(test_dataset)

# In[41]:


dataset['seq_len'] = dataset['seq'].apply(lambda x: len(x))

dataset['seq_chunked'] = dataset['seq'].apply(lambda x : list(chunkstring(x, 510))) #chunk string in segments of 300


# In[42]:


dataset = dataset.iloc[dataset_start:dataset_start+dataset_len]


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
predicted_prbs,gts = [],[]
#print (dataset.iloc[0]['seq_chunked'])

for no_of_index, tokenized_data in enumerate(data_loader):
    
    print(f'sequence {no_of_index}/{len(dataset)}')
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
        continue

        
    model_input_unaltered = tokenized_data['input_ids'].clone()
    tokenized_data['labels'][tokenized_data['labels']==-100] = 0
    inputs = model_input_unaltered.clone()
    

    # First 5 nucleotides we infer from the first 6-mer
    inputs[:, 1:7] = 4 # we mask the first 6 6-mers
    inputs = inputs.to(device) 
    
    if output_logits:
        model_pred = model(inputs)['logits']
    else:
        model_pred = torch.softmax(model(inputs)['logits'], dim=2)
    
    for i in range(5):
        res,gt = extract_prbs_from_pred(prediction=model_pred[0],
                                        pred_pos=1, # first 6-mer (after CLS)
                                        token_pos=i, # we go thorugh first 6-mer
                                        label_pos=i,
                                        label=label)
        predicted_prbs.append(res)
        gts.append(gt)
    
    

    # we do a batched predict to process the rest of the sequence
    predictions = predict_on_batch(tokenized_data, dataset, no_of_index)
    
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

    assert(len(gts) == torch.stack(predicted_prbs).shape[0]), "{} iter, expected len:{} vs actual len:{}".format(no_of_index,
                                                                                   len(gts), 
                                                                                   torch.stack(predicted_prbs).shape[0])

    #XABCDEFGHIJKL -> XABCDE [ABCDEF BCDEFG CDEFGH DEFGHI EFGHIJ FGHIJK] GHIJKL


# In[50]:

print('Saving predictions')

dataset[['seq_name','seq']].drop_duplicates().to_csv(output_dir + f"/seq_{dataset_start}.csv", index=None)


# In[123]:


prbs_arr = np.array(torch.stack(predicted_prbs))

if output_logits:
    output_name = f'logits_{dataset_start}.npy'
else:
    output_name = f'preds_{dataset_start}.npy'

np.save(output_dir + f"/{output_name}", prbs_arr)

