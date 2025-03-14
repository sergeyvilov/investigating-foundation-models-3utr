#!/bin/bash


#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/extract_human.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/extract_human.e

##########################################################
# Extract 3'UTR regions from HAL archive to a FASTA file
# Only Homo Sapiens
#
# Don't take reverse complement for negative genes
# 
#
##########################################################

LINE_WIDTH=80

data_dir=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/

hal_file="$data_dir/600_way/241-mammalian-2020v2.hal"
utr_table="$data_dir/UTR_coords/GRCh38_3_prime_UTR_clean.bed" #see GRCh38_3_prime_UTR_clean.ipynb

output_dir="$data_dir/fasta/"

mkdir -p $output_dir

output_fasta="$output_dir/Homo_sapiens_dna_fwd.fa" 

true > $output_fasta

while read -r chrom human_UTR_start human_UTR_end UTR_ID score strand transcript_ID canonical HGNC_Symbol UTR_len; do

echo  ">$UTR_ID" >> $output_fasta #sequence header

hal2fasta $hal_file Homo_sapiens --sequence $chrom --start $human_UTR_start --length $UTR_len  --lineWidth $LINE_WIDTH| tail -n +2 >> $output_fasta

done <$utr_table

samtools faidx $output_fasta
