import itertools
import numpy as np

import gensim.models 

class Kmerizer:
    '''
    Helper class to generate k-mers and Word2Vec embeddings
    '''
    
    def __init__(self, k):
        
        self.k = k
        
        #generate all possible k-mers, e.g. 
        self.kmers = {"".join(x):i for i,x in zip(range(4**k), itertools.product("ACGT",repeat=k))} 
        
    def kmerize(self, seq):
        '''
        Count all k-mers in the sequence 
        Returns:
        A list with counts corresponding to each possible k-mer from self.kmers
        e.g. for k=2 and seq='ACTAC'
        > [0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        '''
        counts = [0]*4**self.k
        for i in range(len(seq) - self.k + 1): 
            kmer = seq[i:i+self.k]
            counts[self.kmers[kmer]] += 1
        return counts
    
    def tokenize(self, seq):
        '''
        Get all k-mers in the sequence
        Returns:
        A list of all k-mers
        e.g. for 2-mers and seq='ACTAC' 
        > ['AC', 'CT', 'TA', 'AC']
        '''
        kmers = []
        for i in range(len(seq) - self.k + 1): 
            kmer = seq[i:i+self.k]
            kmers.append(kmer)
        return kmers
    
    
def word2vec_model(mpra_df):
    
    '''
    Word2Vec model
    
    k-mers are defined through their context: 
    k-mers with similar context will have similar embeddings
    '''
    
    kmerizer_w2v = Kmerizer(k=4)
    
    w2v_model = gensim.models.Word2Vec(sentences=mpra_df.seq.apply(lambda x: kmerizer_w2v.tokenize(x)), 
                         vector_size=128, window=5, min_count=1, workers=4, sg=1) #default: CBOW

    word2vec_emb = mpra_df.seq.apply(
        lambda x: np.mean([w2v_model.wv[x]  for x in kmerizer_w2v.tokenize(x)],axis=0)) #average embedding of all 4-mers in the sequence

    X = np.stack(word2vec_emb,axis=0)
    
    return X


def minseq_model(mpra_df):
    
    '''
    Minimal sequence model from Griesemer et al. 2021
    
    Extracts following features from mpra_df:
    --nucleotide counts for each base (+4) and maximum among them (+1)
    --dinucleotide counts (+16) and maximum among them (+1)
    --maximum homopolymer length for all bases (+4)
    --maximum dinucleotide repeat length length for all bases (+16)
    --sequence uniformity (+1)
    --Minimal free energy (from RNA fold Software) (+1)
    
    Returns:
    Feature matrix X
    '''
    
    kmerizer1 = Kmerizer(k=1)

    nucl_counts = np.stack(mpra_df.seq.apply(lambda seq: kmerizer1.kmerize(seq)))
    max_nucl_counts_all = nucl_counts.max(axis=1)

    kmerizer2 = Kmerizer(k=2)

    dinucl_counts = np.stack(mpra_df.seq.apply(lambda seq: kmerizer2.kmerize(seq)))
    max_dinucl_counts_all = dinucl_counts.max(axis=1)
    
    max_homopol_len = np.stack(mpra_df.seq.apply(lambda seq: max_repeat_length(seq,1)))
    max_dinucl_len = np.stack(mpra_df.seq.apply(lambda seq: max_repeat_length(seq,2)))

    seq_uniformity = mpra_df.seq.apply(lambda seq: np.sum([seq[i]==seq[i-1] for i in range(1,len(seq))])).values
    #min_free_energy = mpra_df['min_free_energy'].values
    
    X = np.hstack((nucl_counts,np.expand_dims(max_nucl_counts_all,axis=1),
           dinucl_counts,np.expand_dims(max_dinucl_counts_all,axis=1),
           max_homopol_len, max_dinucl_len,
           np.expand_dims(seq_uniformity,axis=1),
           ))
    
    return X

def max_repeat_length(seq,k):
    
    '''
    Get repeat lengths of all k-mers in sequence
    Returns:
    Array of maximal repeat lengths for all k-mers
    '''

    max_subseq_length = {"".join(x):0 for x in itertools.product("ACGT",repeat=k)}

    for start in range(k):
        n_repeats=1 #each subsequence has at least 1 repeat
        for seq_idx in range(start,len(seq)-k+1,k):
            subseq = seq[seq_idx:seq_idx+k] #current subsequence
            if subseq==seq[seq_idx+k:seq_idx+2*k]: #if repeated at the next position
                n_repeats+=1 #increase repeats counter
            else:
                max_subseq_length[subseq] = max(max_subseq_length[subseq],n_repeats) #is the current subseq repeat longer?
                n_repeats = 1 #reinitialize repeats counter
                
        max_subseq_length[subseq] = max(max_subseq_length[subseq],n_repeats) #end of the sequence
       
    repeat_length = np.array(list(max_subseq_length.values()))
    
    return repeat_length