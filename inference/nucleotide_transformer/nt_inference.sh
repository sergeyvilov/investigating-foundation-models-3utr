#!/bin/bash

# USAGE: sbatch --array=0-9%10 predict_3UTR.sh

#SBATCH -o '/lustre/groups/epigenereg01/workspace/projects/vale/MLM/motif_predictions/slurm_logs/%a.o'
#SBATCH -e '/lustre/groups/epigenereg01/workspace/projects/vale/MLM/motif_predictions/slurm_logs/%a.e'
#SBATCH -J NT-pred
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH -c 4 #CPU cores required
#SBATCH -t 2-00:00:00 #Job runtime
#SBATCH --mem=64G

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM'

dataset=$data_dir/'motif_predictions/split_75_25/test.csv'
model_dir=$data_dir/'nucleotide-transform/nucleotide-transformer-v2-500m-multi-species'
output_dir=$data_dir/'motif_predictions/split_75_25/ntrans/NT-MS-v2-500M'

mkdir -p $output_dir

fold=${SLURM_ARRAY_TASK_ID}

python -u nt_inference.py $dataset $model_dir $output_dir $fold





