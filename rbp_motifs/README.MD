Recognition of RBP binding motifs

* Data preprocessing
1. Get PlyloP, CADD, DNABERT, and NT probabilities for all 3'UTR sequences using scripts from [zero-shot-probs](../zero-shot-probs/)
2. Prepprocess e-CLIP data from Van Nostrand et al. using [dataprep/get_eclip](dataprep/get_eclip)
3. Intersect eCLIP and RBNS results and subsample motifs using [dataprep/combine_eclip_RBNS.ipynb](dataprep/combine_eclip_RBNS.ipynb)

* Analysis
1. Add predictions from models using [analysis/add_model_preds.ipynb](analysis/add_model_preds.ipynb)
2. Compute AUC and make plots using [analysis/auc.ipynb](analysis/auc.ipynb)