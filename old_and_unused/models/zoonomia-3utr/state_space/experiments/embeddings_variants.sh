#!/bin/bash

#################################################################
#
#Generate probabilities and embeddings
#
#################################################################

#SBATCH -J selected_emb
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --time=2-00:00:00
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/models/state_space/

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

fasta=$data_dir'variants/selected/variants_rna_1024bp.fa'

logs_dir=$data_dir'variants/embeddings/'

for model in stspace stspace-spaw;do

    checkpoint_dir=$data_dir"models/$model/checkpoints/epoch_252/"

    output_dir="$logs_dir/$model/"

    echo "Output dir: $output_dir"

    if [ $model = 'stspace-spaw' ]; then
      is_species_aware='--species_aware'
    fi

    NN_PARAMETERS="${COMMON_NN_PARAMETERS}  \
    --fasta $fasta  --output_dir ${output_dir} $is_species_aware --checkpoint_dir $checkpoint_dir \
    --test --batch_size 1 --save_probs --mask_at_test 0"

    echo "output dir = ${output_dir}"
    echo "NN parameters = ${NN_PARAMETERS}"

    mkdir -p $output_dir

    python -u main.py ${NN_PARAMETERS} > ${output_dir}/log 2>&1

done
