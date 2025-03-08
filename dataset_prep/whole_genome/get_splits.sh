#!/bin/bash

#SBATCH -J get_splits
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e


data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/whole_genome/'

python get_splits.py
