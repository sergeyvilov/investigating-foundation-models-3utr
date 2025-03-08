# Investigating the performance of foundation models on human 3’UTR sequences

Sergey Vilov and Matthias Heinig

[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.02.09.579631v1)

## Codes for data preprocessing and analysis

* [rbp_motifs](rbp_motifs/) : evaluate the models on RBP binding motifs prediction (TASK 1)
* [variant_effect](variant_effect/) : evaluate the models on variants from ClinVar, gnomAD, eQTL, and CADD (TASK 2)
* [mpra](mpra/) : prediction of MPRA activity from (Griesemer et al., 2021) and (Siegel et al., 2022) (TASK 3)
* [half_life](half_life/) : prediction of mRNA half-life from (Agarwal and Kelley, 2022) (TASK 4)
* [dataset_prep](dataset_prep/) : build the multispecies dataset from Zoonomia whole genome alignment
* [embeddings](embeddings/) : generate embeddings for the DNABERT, DNABERT-2, NT as well as embeddings and per-base zero-shot scores for StateSpace models
* [zero-shot-probs](zero-shot-probs/) : derive per-base zero-shot scores for DNABERT, NT, PhyloP, and CADD models

The analysis data, scores for all models, and model weights can be found in our [Zenodo repository](https://zenodo.org/records/10655595)

## Links to the scripts used to generate paper figures and tables:

[Fig. 1: ROC AUC scores for RBP binding motif predictions](rbp_motifs/analysis/auc.ipynb)

[Fig. 2: ROC curves for prediction of proxy-functional variants on ClinVar, gnomAD, eQTL, and CADD data using the best predictor for each model](variant_effect/analysis/auc.ipynb)

[Fig. 3: Pearson r correlation coefficient between mRNA half-life prediction and ground truth data from (Agarwal and Kelley, 2022)](half_life/regression/analyse.ipynb)

[Fig. S1: Distribution of 3’UTR length for 18,134 transcripts of the human genome](dataset_prep/3utr/unaligned/analysis/plot_3UTR.ipynb)

[Fig. S2: Pearson r correlation between per-nucleotide probabilities predicted by each model and the ground truth probability for the Zoonomia dataset (Zoo-AL)](rbp_motifs/analysis/pref.ipynb)

[Fig. S3: Difference between ROC AUC scores based on the variant influence score (VIS) and the reference allele probability (pref), as a function of the maximum window W around the variant used to compute VIS](variant_effect/analysis/vis_vs_distance.ipynb)

[Table 1: Pearson r correlation coefficient between Ridge-based predictions from sequence embeddings and ground truth MPRA expression from (Griesemer et al., 2021)](mpra/mpra_griesemer_2021/regression/analyse.ipynb)

[Table S2: ROC AUC scores for RBP binding motif predictions, for all motifs, proxy-functional motifs within the top 10% conservation, proxy-functional motifs within the bottom 10% conservation, as predicted by PhyloP-241way](rbp_motifs/analysis/auc.ipynb)

[Table S3: ROC AUC scores for ClinVar, gnomAD, eQTL, and CADD data computed based on zero-shot functionality scores for all models](variant_effect/analysis/auc.ipynb)

[Table S4: ROC AUC scores from MLP-based prediction of proxy-functional variants on ClinVar, gnomAD, eQTL, and CADD data using language model embeddings](variant_effect/analysis/auc.ipynb)

[Table S5: ROC AUC scores from prediction of proxy-functional variants on ClinVar, gnomAD, eQTL, and CADD data using alignment-based models](variant_effect/analysis/auc.ipynb)

[Table S6: Pearson r correlation coefficient between SVR-based predictions from sequence embeddings and ground truth MPRA activity from (Griesemer et al., 2021)](mpra/mpra_griesemer_2021/regression/analyse.ipynb)

[Table S7: Pearson r correlation coefficient between Ridge-based predictions from sequence embeddings and ground truth MPRA data from (Siegel et al., 2022)](mpra/mpra_siegel_2022/regression/analyse.ipynb)

[Table S8: Pearson r correlation coefficient between SVR-based predictions from sequence embeddings and ground truth MPRA data from (Siegel et al., 2022)](mpra/mpra_siegel_2022/regression/analyse.ipynb)

[Table S9: Pearson r correlation coefficient between mRNA half-life prediction and ground truth data from (Agarwal and Kelley, 2022), using different 3’UTR embeddings](half_life/regression/analyse.ipynb)

## Installation

1. Create new [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```
conda create -n lm-3utr-models python=3.10
conda activate lm-3utr-models
```
2. Install [Pytorch v.2.0.1](https://pytorch.org/)

3. Install the other requirements using pip:

```
pip install -r requirements.txt
```

4. To train DNABERT-2 models also install 
```
pip install triton==2.0.0.dev20221202 --force --no-dependencies
```

Training of DNABERT-2 is currently only possible on NVIDIA A100 due to the employed flash attention implementation.