#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM'

fasta=$data_dir'/siegel_2022/variants/fasta/variants_dna_fwd.fa'

#for model_name in 'DNABERT' 'DNABERT-2' 'NT-MS-v2-500M'; do
for model_name in 'NT-MS-v2-500M'; do
    for fold in {0..9}; do
    output_dir="$data_dir/siegel_2022/variants/embeddings/dna_fwd/$model_name/"
    srun -p gpu_p --qos=gpu_normal -o logs/siegel-$model_name-$fold.o -e logs/siegel-$model_name-$fold.e --nice=10000 -J $model_name -c 10 --mem=64G --gres=gpu:1 --time=2-00:00:00 \
    python gen_embeddings.py --N_folds 10 --fold $fold --fasta $fasta --model $model_name --output_dir $output_dir &
    done
done
