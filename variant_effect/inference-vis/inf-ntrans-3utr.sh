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
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/zero-shot-probs/nucleotide_transformer

source  ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm'

model_name='ntrans-v2-100m-3utr-2e'

fasta=$data_dir'/variants/selected/variants_rna.fa'
checkpoint_dir=$data_dir"/models/zoonomia-3utr/$model_name/checkpoints/chkpt_633/"

strand_bed=$data_dir'/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

whitelist=$data_dir'/variants/selected/vis_whitelist.tsv'

fold=${SLURM_ARRAY_TASK_ID:-0}
#N_folds=${SLURM_ARRAY_TASK_COUNT:-0}
N_folds=10

for masking in false ; do

    for ref_aware in true ; do

        output_dir="$data_dir/variants/variant_influence_score/$model_name/"

        mkdir -p $output_dir

        if [ ! -f "$output_dir/predictions_$fold.pickle" ]; then
        srun python -u nt_inference.py --fasta $fasta --checkpoint_dir $checkpoint_dir --output_dir $output_dir  --ref_aware ${ref_aware} --whitelist $whitelist --N_folds $N_folds --fold $fold  --masking $masking --strand_bed $strand_bed > $output_dir/log_$fold 2>&1
        fi

    done

done
