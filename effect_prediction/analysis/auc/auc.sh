#!/bin/bash

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/variants/'

output_dir=$data_dir/auc

mkdir -p $output_dir

utr_tsv=$data_dir'/model_scores_snp.tsv'

#for split in clinvar gnomAD eQTL-GRASP eQTL-susie; do
for split in gnomAD eQTL-susie; do

    srun -J $split -p cpu_p --qos=cpu_normal --time=2-00:00:00 --mem=64G -o logs/$split.o -e logs/$split.e \
    python -u auc.py $utr_tsv $split $output_dir/$split.tsv &

done
