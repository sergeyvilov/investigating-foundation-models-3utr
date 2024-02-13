#!/bin/bash

# USAGE: sbatch --array=0-9%10 dnabert.sh

#SBATCH -o '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o'
#SBATCH -e '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e'
#SBATCH -J dnbinf
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=GPU_Nvidia_Tesla_A100
#SBATCH -c 4 #CPU cores required
#SBATCH -t 2-00:00:00 #Job runtime
#SBATCH --mem=64G

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib


data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

fasta=$data_dir/'fasta/Homo_sapiens_rna.fa'
checkpoint_dir=$data_dir/'models/dnabert/6-new-12w-0'
output_dir=$data_dir/'human_3utr/probs/dnabert/'

mkdir -p $output_dir

fold=${SLURM_ARRAY_TASK_ID}

python -u dna_bert_eval.py $fasta $checkpoint_dir $output_dir $fold





