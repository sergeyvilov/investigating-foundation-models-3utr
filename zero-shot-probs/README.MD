Get zero-shot probabilities with DNABERT, NT, CADD and PhyloP models

Needed for computing zero-shot scores in [RBP motifs](../rbp_motifs) and in [variant effect](../variant_effect).

For the StateSpace models use [embeddings/state_space](../embeddings/state_space) to generate both embeddings as well as per-nucleotide probabilities.

The scores are computed for all positions in each of the 18,134 3'UTR sequences.
When the sequnece is longer than the model's field of view, it is split into chunks after tokenization.

We use fasta/Homo_sapiens_rna.fa for RNA-based models: DNABERT-3UTR,NTv2-100M-3UTR,DNABERT2-3UTR,StateSpace, StateSpace-SA

We use fasta/Homo_sapiens_dna_fwd.fa for DNA-based models: 
DNABERT,NT-MS-v2-100M,StateSpace-DNA,StateSpace-SA-DNA,CADD-1.7,PhyloP-241way,PhyloP-241way)

For all models, the resulting probabilities are saved in pickle files with the sequences matching those in the input FASTA. 
