#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

batch_size=30

for model_name in 'dnabert' 'dnabert2' 'ntrans-v2-500m' 'ntrans-v2-250m' 'dnabert-3utr' 'dnabert2-3utr' 'ntrans-v2-250m-3utr'; do
    if [[ $model_name =~ "dnabert2" ]]; then
       constraint="--constraint=GPU_Nvidia_Tesla_A100"
    else
       constraint=""
    fi
    if [[ $model_name =~ "-3utr" ]]; then
       fasta=$data_dir'/mpra/siegel_2022/fasta/variants_rna.fa'
    else
       fasta=$data_dir'/mpra/siegel_2022/fasta/variants_dna_fwd.fa'
    fi
    output_dir="$data_dir/mpra/siegel_2022/embeddings/$model_name/"
    mkdir -p $output_dir
    srun -p gpu_p --qos=gpu_normal $constraint -o logs/$model_name-sieg.o -e logs/$model_name-sieg.e --nice=10000 -J $model_name-sieg -c 4 --mem=64G --gres=gpu:1 --time=2-00:00:00 \
    python -u gen_embeddings.py --fasta $fasta --model $model_name --output_dir $output_dir --batch_size $batch_size &
done
