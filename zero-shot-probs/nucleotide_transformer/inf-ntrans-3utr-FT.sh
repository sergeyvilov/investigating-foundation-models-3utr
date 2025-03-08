#!/bin/bash

#SBATCH -o '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o'
#SBATCH -e '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e'
#SBATCH -J ntrans-3utr
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH -c 4 #CPU cores required
#SBATCH -t 1-00:00:00 #Job runtime
#SBATCH --mem=64G

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

model_name='ntrans-v2-100m-3utr-f'

fasta=$data_dir/'fasta/Homo_sapiens_rna.fa'
checkpoint_dir=$data_dir"/models/zoonomia-3utr/$model_name/checkpoints/chkpt_61/"

output_dir=$data_dir"/human_3utr/probs/$model_name"
mkdir -p $output_dir

fold=${SLURM_ARRAY_TASK_ID:-0}
#N_folds=${SLURM_ARRAY_TASK_COUNT:-0}
N_folds=10

srun python -u nt_inference.py --fasta $fasta --checkpoint_dir $checkpoint_dir --output_dir $output_dir --N_folds $N_folds --fold $fold  --masking true --ref_aware false > $output_dir/log_$fold 2>&1
