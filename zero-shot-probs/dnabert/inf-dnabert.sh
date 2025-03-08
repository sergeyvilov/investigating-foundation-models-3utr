#!/bin/bash

# USAGE: sbatch --array=0-20%10 dnabert.sh

#SBATCH -o '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o'
#SBATCH -e '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e'
#SBATCH -J dnbinf
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH -x supergpu14
#SBATCH -c 4 #CPU cores required
#SBATCH -t 1-00:00:00 #Job runtime
#SBATCH --mem=64G

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

model_name='dnabert'

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

fasta=$data_dir/'fasta/Homo_sapiens_dna_fwd.fa'
checkpoint_dir=$data_dir"/models/whole_genome/$model_name/6-new-12w-0"

output_dir=$data_dir"/human_3utr/probs/$model_name/"
mkdir -p $output_dir

fold=${SLURM_ARRAY_TASK_ID:-0}
#N_folds=${SLURM_ARRAY_TASK_COUNT:-0}
N_folds=20

srun python -u dna_bert_eval.py --fasta $fasta --checkpoint_dir $checkpoint_dir --output_dir $output_dir --N_folds $N_folds --fold $fold > $output_dir/log_$fold 2>&1

