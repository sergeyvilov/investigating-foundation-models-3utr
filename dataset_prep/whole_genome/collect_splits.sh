#!/bin/bash

##########################################################
# Collect shuffled splits based on shuffled_idx
#
#
#TO RUN: sbatch --array=0-999%30 collect_splits.sh
##########################################################

#SBATCH -J collect_splits
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e

workdir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/whole_genome/'

cd $workdir

split_idx=${SLURM_ARRAY_TASK_ID}

output_fa=shuffled_splits/split_${split_idx}.fa
input_txt=shuffled_splits/split_${split_idx}.txt

samtools faidx $workdir/all_species.fa -r $input_txt > $output_fa
