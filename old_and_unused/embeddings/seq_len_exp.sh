#!/bin/bash

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

batch_size=30

model_name='ntrans-v2-100m-3utr-2e'

if [[ $model_name =~ "dnabert2" ]]; then
    constraint='--constraint=a100_80gb|a100_40gb'
else
    constraint=""
fi

if [[ $model_name =~ "-3utr" ]]; then
    fasta=$data_dir'/fasta/Homo_sapiens_rna.fa'
else
    fasta=$data_dir'/fasta/Homo_sapiens_dna_fwd.fa'
fi

for max_seq_len in 512 1024 2048 4096; do

    output_dir="$data_dir/half_life/agarwal_2022/seqlen-exp/embeddings/$model_name/$max_seq_len/"
    
    mkdir -p $output_dir
    
        srun -p gpu_p --qos=gpu_normal $constraint -o $output_dir/log.o -e $output_dir/log.e --nice=10000 -J $model_name-$max_seq_len -c 4 --mem=20G --gres=gpu:1 --time=1-00:00:00 \
        python -u gen_embeddings.py --fasta $fasta --model $model_name --max_seq_len $max_seq_len --output_dir $output_dir --batch_size $batch_size &
        
done
