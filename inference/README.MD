Get reference allele probabilities with [DNABERT](dnabert/), [NT](nucleotide_transformer/), and [PhyloP](PhyloP/) model

For PhyloP probabilities equal the conservation score

For genome-wide DNABERT,NT and PhyloP models, the inference is always performed on the forward DNA strand. For genes on the negative strand, the probabilities are then reverse complemented to match the mRNA content. For 3'UTR-specific models, e.g. DNABERT-3UTR and NTv2-250M-3UTR, the inference is performed directly on RNA sequences.