#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM'

fasta=$data_dir'/griesemer/fasta/GRCh38_UTR_variants_no_reverse.fa'

for model_name in 'DNABERT' 'DNABERT-2' 'NT-MS-v2-500M'; do
    output_dir="$data_dir/griesemer/embeddings/$model_name/"
    srun -p gpu_p --qos=gpu_normal -o logs/griesman-$model_name.o -e logs/griesman-$model_name.e --nice=10000 -J $model_name -c 10 --mem=64G --gres=gpu:1 --time=2-00:00:00 \
    python gen_embeddings.py $fasta $model_name $output_dir &
done