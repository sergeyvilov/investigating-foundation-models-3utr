Variant effect prediction based on functionality scores for ClinVar, gnomAD, and eQTL variants

* [analysis](analysis/) : scripts to compute ROC AUC, odds ratios and plot distributions of single-position zero-shot scores
* [classify](classify/) : train and evaluate a MLP to predict functional variants
* [dataprep](dataprep/) : collect variants from different sources, combine them together and generate a .fasta file with [add_model_scores.ipynb](dataprep/add_model_scores.ipynb), perform [inference](../inference) and generate [embeddings](../embeddings), then combine zero-shot scores from different models with [combine_sources.ipynb](dataprep/combine_sources.ipynb)

