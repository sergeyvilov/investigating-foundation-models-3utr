#!/bin/bash

##########################################################
# Extract all genomes from HAL archive to a FASTA file
#
#
#TO RUN: sbatch --array=1-241%10 extract_train.sh
##########################################################

#SBATCH -J hal2fasta
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e


output_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/whole_genome/species/'

hal_file='/lustre/groups/epigenereg01/workspace/projects/vale/vae_effect_prediction/data/605-vertebrate-2020/605-vertebrate-2020.hal'

species_list='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_species.txt'

mkdir -p $output_dir

max_species=$(wc -l $species_list|cut -d" " -f1) #=241

if [ "${SLURM_ARRAY_TASK_ID}" -le "$max_species" ]; then

    species_name=$(head -n ${SLURM_ARRAY_TASK_ID} $species_list | tail -1)

    hal2fasta $hal_file $species_name |sed "s/>/>$species_name:/" > $output_dir/$species_name.fa #add pecies name to the contig name

    #samtools faidx $output_dir/$species_name.fa

fi
