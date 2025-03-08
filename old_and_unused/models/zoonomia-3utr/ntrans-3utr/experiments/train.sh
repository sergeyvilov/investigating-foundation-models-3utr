#!/bin/bash

#################################################################
#
#Train nucleotide transformer on a single GPU
#
#################################################################

#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --constraint=GPU_Nvidia_Tesla_A100
#SBATCH -x gpusrv[26-35,38-52]
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/models/ntrans-3utr/

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

model_name=$SLURM_JOB_NAME

fasta='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_mammals.shuffled.fa'

output_dir="/lustre/groups/epigenereg01/workspace/projects/vale/mlm/models/ntrans-${model_name}-3utr/"

#model_checkpoint="/lustre/groups/epigenereg01/workspace/projects/vale/mlm/dnabert2-3utr/single_gpu/checkpoints/after_2days/epoch_28"

if [ ! -z "${model_checkpoint}" ]; then
	checkpoint_dir="--checkpoint_dir ${model_checkpoint}"
fi


echo "Output dir: $output_dir"

NN_PARAMETERS="${COMMON_NN_PARAMETERS}  \
--fasta $fasta  --output_dir ${output_dir} ${checkpoint_dir} \
--save_at -1 2:100:2 --validate_every 0 --model_name ${model_name}   --species_agnostic \
--train_chunks 64 --tot_epochs 100"

echo "output dir = ${output_dir}"
echo "NN parameters = ${NN_PARAMETERS}"

mkdir -p $output_dir

python -u main.py ${NN_PARAMETERS} > ${output_dir}/log 2>&1
