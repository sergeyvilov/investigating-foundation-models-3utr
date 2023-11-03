#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM'

fasta=$data_dir'/agarwal_2022/Homo_sapiens_no_reverse_half_life.fa'

N_folds=10

batch_size=10

for model_name in 'DNABERT-2'; do
    for fold in {0..9}; do
    
    #exclude_list="$data_dir/3UTR_embeddings/$model_name/processed_utrs.csv"
    output_dir="$data_dir/3UTR_embeddings/$model_name/"
    
    srun -p gpu_p --qos=gpu_normal -o "logs/$model_name-fold_$fold.o" -e "logs/$model_name-fold_$fold.e" --nice=10000 -J "$model_name-$fold" -c 10 --mem=64G --gres=gpu:1 --time=2-00:00:00 \
    python -u gen_embeddings.py --fasta $fasta --model $model_name --output_dir $output_dir --batch_size $batch_size --N_folds $N_folds --fold $fold &
    done
done
