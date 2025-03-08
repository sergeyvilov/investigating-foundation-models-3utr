#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

#include_txt="${data_dir}/half_life/agarwal_2022/regions.tsv" #only regions used in agarwal et al.

batch_size=30

for model_name in 'dnabert' 'dnabert-3utr-2e' 'dnabert2' 'dnabert2-zoo' 'dnabert2-3utr-2e' 'ntrans-v2-100m' 'ntrans-v2-100m-3utr-2e'; do


    if [[ $model_name =~ "-3utr" ]]; then
       fasta=$data_dir'/fasta/Homo_sapiens_rna.fa'
    else
       fasta=$data_dir'/fasta/Homo_sapiens_dna_fwd.fa'
    fi
    output_dir="$data_dir/human_3utr/embeddings/$model_name/"

    if [[ $model_name =~ "dnabert2" ]]; then
       constraint='--constraint=a100_80gb|a100_40gb|a100_20gb'
    else
       constraint=""
    fi

    mkdir -p $output_dir
    srun -p gpu_p --qos=gpu_normal $constraint -o ${output_dir}/log.o -e ${output_dir}/log.e --nice=10000 -J $model_name-human3utr -c 4 --mem=64G --gres=gpu:1 --time=1-00:00:00 \
    python -u gen_embeddings.py --fasta $fasta --model $model_name --output_dir $output_dir --batch_size $batch_size &
done
