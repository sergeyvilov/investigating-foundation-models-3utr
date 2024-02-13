#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

#include_txt="${data_dir}/half_life/agarwal_2022/regions.tsv" #only regions used in agarwal et al.

batch_size=30

#for model_name in 'dnabert' 'dnabert2' 'ntrans-v2-500m' 'ntrans-v2-250m' 'dnabert-3utr' 'dnabert2-3utr' 'ntrans-v2-250m-3utr'; do
for model_name in 'dnabert' 'dnabert-3utr' 'dnabert2' 'dnabert2-3utr'; do
    if [[ $model_name =~ "dnabert2" ]]; then
       constraint="--constraint=GPU_Nvidia_Tesla_A100"
    else
       constraint=""
    fi
    if [[ $model_name =~ "-3utr" ]]; then
       fasta=$data_dir'/fasta/Homo_sapiens_rna.fa'
    else
       fasta=$data_dir'/fasta/Homo_sapiens_dna_fwd.fa'
    fi
    output_dir="$data_dir/human_3utr/embeddings/$model_name/"
    mkdir -p $output_dir
    srun -p gpu_p --qos=gpu_normal $constraint -o logs/$model_name-human3utr.o -e logs/$model_name-human3utr.e --nice=10000 -J $model_name-human3utr -c 4 --mem=64G --gres=gpu:1 --time=2-00:00:00 \
    python -u gen_embeddings.py --fasta $fasta --model $model_name --output_dir $output_dir --batch_size $batch_size &
done
