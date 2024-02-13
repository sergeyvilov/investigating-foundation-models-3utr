#  Investigating the performance of foundation models on human 3’UTR regions

Sergey Vilov and Matthias Heinig

Codes for data analysis:

* [fasta_prep](fasta_prep/) : extract 3'UTR sequences from Zoonomia whole genome alignment
* [effect_prediction](effect_prediction/) : compute functionality scores for ClinVar, gnomAD, and eQTL variants and evaluate the models
* [motif_search](motif_search/) : evaluate how well the models can predict RBP binding motifs
* [half_life](half_life/) : prediction of mRNA half-life from Agarwal and Kelley, 2022 et based on language model embeddings
* [mpra](mpra/) : prediction of measured MPRA activity for Griesemer et al., 2021 and Siegel et al., 2022 experiments
* [embeddings](embeddings/) : generation of embeddings with DNABERT, DNABERT-2, and NT-MS-v2-500M models
* [inference](inference/) : derive per-base scores for DNABERT, Nucleotide Transformer, and PhyloP models
