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
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e

workdir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_mammals/'

cd $workdir

split_idx=${SLURM_ARRAY_TASK_ID}

output_fa=shuffled_split/fa/split_${split_idx}.fa
input_fai=shuffled_split/fai/split_${split_idx}.fai

> $output_fa

while  read -r seq_name _ ;do

  species=$(echo $seq_name|cut -d":" -f2|sed -E 's/([^_]+)_([^_]+).*/\1_\2/')

  fasta=species/$species.fa

  samtools faidx $fasta "$seq_name" >> $output_fa

done < $input_fai
