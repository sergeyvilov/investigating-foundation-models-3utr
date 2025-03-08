#!/bin/bash

# USAGE: sbatch --array=0-9%10 dnabert-3utr-2e.sh

#SBATCH -o '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o'
#SBATCH -e '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e'
#SBATCH -J dnbinf-3utr
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH -x supergpu14
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb|a100_40gb
#SBATCH -x gpusrv[26,28-35,38-52]
#SBATCH -c 4 #CPU cores required
#SBATCH -t 1-00:00:00 #Job runtime
#SBATCH --mem=64G
#SBATCH -x gpusrv66
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/zero-shot-probs/dnabert

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib


model_name='dnabert-3utr-2e'

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

fasta=$data_dir'/variants/selected/variants_rna.fa'
checkpoint_dir=$data_dir/"models/zoonomia-3utr/$model_name/checkpoints/chkpt_40/" #6 epochs

strand_bed=$data_dir'/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

whitelist=$data_dir'/variants/selected/reference_allele.tsv'

output_dir="$data_dir/variants/zero-shot-probs/$model_name/"

mkdir -p $output_dir

fold=${SLURM_ARRAY_TASK_ID:-0}
#N_folds=${SLURM_ARRAY_TASK_COUNT:-0}
N_folds=10

if [ ! -f "$output_dir/predictions_$fold.pickle" ]; then
    srun python -u dna_bert_eval.py --fasta $fasta --checkpoint_dir $checkpoint_dir --output_dir $output_dir --N_folds $N_folds --fold $fold --whitelist $whitelist --strand_bed $strand_bed --predict_only_lowercase --crop_lowercase > $output_dir/log_$fold 2>&1
fi
