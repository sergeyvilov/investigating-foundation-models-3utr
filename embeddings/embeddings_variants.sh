#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM'

fasta=$data_dir'/perbase_pred/fasta/variants-510bp.fa'

batch_size=30

#for model_name in 'DNABERT' 'DNABERT-2' 'NT-MS-v2-500M'; do
for model_name in  'DNABERT'; do
    output_dir="$data_dir/perbase_pred/embeddings_scores/$model_name/"
    mkdir -p $output_dir
    srun -p gpu_p --qos=gpu_normal -o logs/$model_name-vars.o -e logs/$model_name-vars.e --nice=10000 -J $model_name -c 10 --mem=64G --gres=gpu:1 --time=2-00:00:00 \
    python -u gen_embeddings.py $fasta $model_name $output_dir $batch_size &
done
