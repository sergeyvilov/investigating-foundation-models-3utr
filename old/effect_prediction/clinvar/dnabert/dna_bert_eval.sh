#!/bin/bash

#################################################################
#
#DNA BERT inference
#
#sbatch --array=0-5%10 dna_bert_eval.sh
#################################################################

#SBATCH -J DNABERT
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/motif_predictions/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/motif_predictions/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-mlm

model=default

output_dir="/s/project/mll/sergey/effect_prediction/MLM/clinvar/dnabert/$model/preds/"

test_dataset='/s/project/mll/sergey/effect_prediction/MLM/clinvar/dnabert/clinvar_refseq.csv'

dataset_len=500

output_logits=1

total_seq=$(wc -l $test_dataset|cut -d" " -f1)

mkdir -p $output_dir

#python -u dna_bert_eval.py  $output_dir $test_dataset > ${output_dir}/log 2>${output_dir}/err

c=0

for dataset_start in `seq 0 $dataset_len $total_seq`; do

    if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

        python -u dna_bert_eval.py  $output_dir $test_dataset $dataset_start $dataset_len $output_logits > ${output_dir}/log_$dataset_start 2>${output_dir}/err_$dataset_start 

    fi

    c=$((c+1))

done
