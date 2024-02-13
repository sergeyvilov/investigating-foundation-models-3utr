#!/bin/bash

# USAGE: sbatch --array=0-9%10 nt_inference.sh

#SBATCH -o '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o'
#SBATCH -e '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e'
#SBATCH -J nt-probs
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH -c 4 #CPU cores required
#SBATCH -t 2-00:00:00 #Job runtime
#SBATCH --mem=64G

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

fasta=$data_dir/'fasta/Homo_sapiens_rna.fa'
checkpoint_dir=$data_dir/'models/ntrans-v2-250m-3utr/checkpoints/epoch_23/'
output_dir=$data_dir/'human_3utr/probs/ntrans-v2-250m-3utr/'

mkdir -p $output_dir

fold=${SLURM_ARRAY_TASK_ID}

python -u nt_inference.py $fasta $checkpoint_dir $output_dir $fold





