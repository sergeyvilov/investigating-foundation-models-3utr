#!/bin/bash

#################################################################
#
#Collect sequences for each FASTA subfile
#
#TO RUN: sbatch --array=0-753%10 collect_sequences.sh
#################################################################

#SBATCH -J collect_splits
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/shuffled_split/logs/%A_%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/shuffled_split/logs/%A_%a.e

workdir='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals'

cd $workdir

split_idx=${SLURM_ARRAY_TASK_ID}

output_fa=shuffled_split/fa/split_${split_idx}.fa
input_fai=shuffled_split/fai/split_${split_idx}.fai

> $output_fa

while  read -r seq_name _ ;do

  species=$(echo $seq_name|cut -d":" -f2|sed -E 's/([^_]+)_([^_]+).*/\1_\2/')

  fasta=species/$species.fa

  samtools faidx $fasta "$seq_name" >> $output_fa

done < <(grep -v Homo_sapiens $input_fai)
