# Investigating the performance of foundation models on human 3’UTR sequences

Sergey Vilov and Matthias Heinig

[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.02.09.579631v1)

## Codes for data analysis

* [effect_prediction](effect_prediction/) : compute functionality scores for ClinVar, gnomAD, and eQTL variants and evaluate the models
* [motif_search](motif_search/) : evaluate the models on RBP binding motifs prediction
* [half_life](half_life/) : prediction of mRNA half-life from (Agarwal and Kelley, 2022) based on language model embeddings
* [mpra](mpra/) : prediction of measured MPRA activity for (Griesemer et al., 2021) and (Siegel et al., 2022) experiments
* [embeddings](embeddings/) : generate embeddings for the DNABERT, DNABERT-2, and NT models
* [inference](inference/) : derive per-base scores for DNABERT, NT, and PhyloP models

**The codes for extraction of 3'UTR sequences from the Zoonomia .hal alignment and the scripts for model training will be made available upon the paper acceptance.**

The intermediate data for the downstream tasks can be found at [https://zenodo.org/records/10655595](https://zenodo.org/records/10655595). The 3'UTR multispecies fasta files and model weights will be added to the Zenodo repository upon the paper acceptance.

## Links to the scripts used to generate paper figures and tables:

[Fig. 1: Odds Ratios and mobility distribution for RBP binding sites recognition](motif_search/plot_odds.ipynb)

[Fig. 2: ROC curves for embeddings-based variant effect predictions on ClinVar, gnomAD, and eQTL data](effect_prediction/analysis/auc/auc.ipynb)

[Fig. S1: Distribution of 3’UTR length for 18,134 transcripts of the human genome.](fasta_prep/unaligned/plot_3UTR.ipynb)

[Fig. S2: Average mobility for the putative functional motifs at 425,413 positions as a function of the conservation distance R.](motif_search/mobility/plots.ipynb)

[Table 1: Pearson r correlation coefficient between Ridge-based predictions from sequence embeddings and ground truth MPRA expression from (Griesemer et al., 2021).](mpra/mpra_griesemer_2021/regression/analyse.ipynb)

[Table S1: ROC AUC scores for ClinVar, gnomAD, and eQTL data computed based on zero-shot functionality scores for all models.](effect_prediction/analysis/auc/auc.ipynb)

[Table S2: ROC AUC scores from prediction of functional variants on ClinVar, gnomAD, and eQTL data using language model embeddings and PhyloP conservation scores.](effect_prediction/analysis/auc/auc.ipynb)

[Table S3: Pearson r correlation coefficient between SVR-based predictions from sequence embeddings and ground truth MPRA activity from (Griesemer et al., 2021).](mpra/mpra_griesemer_2021/regression/analyse.ipynb)

[Table S4: Pearson r correlation coefficient between Ridge-based predictions from sequence embeddings and ground truth MPRA data from (Siegel et al., 2022).](mpra/mpra_siegel_2022/regression/analyse.ipynb)

[Table S5: Pearson r correlation coefficient between SVR-based predictions from sequence embeddings and ground truth MPRA data from (Siegel et al., 2022).](mpra/mpra_siegel_2022/regression/analyse.ipynb)

[Table S6: Pearson r correlation coefficient between mRNA half-life prediction and ground truth data from (Agarwal and Kelley, 2022), using different 3’UTR embeddings.](half_life/regression/analyse.ipynb)

[Table S7: Pearson r correlation coefficient for mRNA half-life prediction with the BC3MS model based on different 3’UTR embeddings and the Saluki model.](half_life/regression/analyse.ipynb)

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
