#!/bin/bash

#################################################################
#
#Default run of species-agnostic model
#
#################################################################

#SBATCH -J stspace
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --constraint=GPU_Nvidia_Tesla_A100
#SBATCH -x gpusrv[26,28-35,38-52]
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/models/state_space/

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

fasta=$data_dir'fasta/241_mammals.shuffled.fa'

logs_dir=$data_dir'models/'

#model_checkpoint="/lustre/groups/epigenereg01/workspace/projects/vale/mlm/state_space/single_gpu/$model/checkpoints/after_2days/epoch_12"

output_dir="$logs_dir/stspace/"

echo "Output dir: $output_dir"

if [ ! -z "${model_checkpoint}" ]; then
	checkpoint_dir="--checkpoint_dir ${model_checkpoint}"
fi

md5=($(tail $fasta|md5sum))
basename_fa=$(basename $fasta .fa)
local_fasta=/localscratch/sergey.vilov/$basename_fa-$md5.fa

if [ ! -f "${local_fasta}" ]; then
    echo $local_fasta not found, copying $fasta
		mkdir -p /localscratch/sergey.vilov/
    cp $fasta.fai $local_fasta.fai
    cp $fasta $local_fasta
else
    md5_local=($(tail $local_fasta|md5sum))
    if [ $md5 != $md5_local ];then
        echo "md5 doesn't match, copying can be in progress..."
        echo "exiting..."
        exit 1
    fi
fi

NN_PARAMETERS="${COMMON_NN_PARAMETERS}  \
--fasta $local_fasta  --output_dir ${output_dir} \
--save_at -1 8:3000:8 --validate_every 8  $checkpoint_dir \
--train_splits 16 --tot_epochs 3000 --n_layers 4 --batch_size 64 --validate_every 0 --weight_decay 0 --seq_len 5000 --d_model 128"

echo "output dir = ${output_dir}"
echo "NN parameters = ${NN_PARAMETERS}"

mkdir -p $output_dir

python -u main.py ${NN_PARAMETERS} > ${output_dir}/log 2>&1
