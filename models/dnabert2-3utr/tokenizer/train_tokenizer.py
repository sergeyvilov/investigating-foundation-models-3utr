import pandas as pd
from transformers import PreTrainedTokenizerFast
import pysam
from tqdm import tqdm

workdir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/'

all_fa = workdir + '241_mammals.shuffled.fa'

seq_names = pd.read_csv(all_fa + '.fai', sep='\t', header=None, usecols=[0])[0].squeeze().values

all_fasta = pysam.FastaFile(all_fa)

seq_names = seq_names[2000000:2200000]

def seq_gen():
    for seq_idx,seq_name in enumerate(seq_names):
        seq=all_fasta.fetch(seq_name).upper().replace('-','')
        if seq_idx%1000==0:
            print(f'seq {seq_idx+1} out of {len(seq_names)}')
        yield(seq)


from tokenizers import SentencePieceBPETokenizer

tokenizer = SentencePieceBPETokenizer(unk_token = '[UNK]')
tokenizer.train_from_iterator(
    seq_gen(),
    vocab_size=4096,
    min_frequency=1,
    show_progress=True,
    length=len(seq_names),
)

from transformers import PreTrainedTokenizerFast

transformer_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer
)

transformer_tokenizer.save_pretrained('./multispecies_tokenizer')
