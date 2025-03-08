#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

batch_size=30

for model_name in 'dnabert' 'dnabert-3utr-2e' 'dnabert2' 'dnabert2-zoo' 'dnabert2-3utr-2e' 'ntrans-v2-100m' 'ntrans-v2-100m-3utr-2e'; do

    if [[ $model_name =~ "-3utr" ]]; then
       fasta=$data_dir'/mpra/siegel_2022/fasta/variants_rna.fa'
    else
       fasta=$data_dir'/mpra/siegel_2022/fasta/variants_dna_fwd.fa'
    fi

    if [[ $model_name = "dnabert" ]] || [[ $model_name = "dnabert-3utr-2e" ]]; then
       crop_options="--max_seq_len 508 --crop_lowercase"
    else
       crop_options=""
    fi

    if [[ $model_name =~ "dnabert2" ]]; then
       constraint='--constraint=a100_80gb|a100_40gb|a100_20gb'
    else
       constraint=""
    fi
    
    output_dir="$data_dir/mpra/siegel_2022/embeddings/$model_name/"
    
    mkdir -p $output_dir
    
    srun -p gpu_p --qos=gpu_normal $constraint -o ${output_dir}/log.o -e ${output_dir}/log.e --nice=10000 -J $model_name-sieg -c 4 --mem=64G --gres=gpu:1 --time=1-00:00:00 \
    python -u gen_embeddings.py --fasta $fasta --model $model_name --output_dir $output_dir --batch_size $batch_size ${crop_options} &
done
