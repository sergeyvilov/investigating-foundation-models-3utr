#!/bin/bash

#SBATCH -J Extract_MSA
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --partition=cpu_p
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000
#SBATCH --qos cpu_normal
#SBATCH -x cpusrv[18-28]
#SBATCH --output=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.out
#SBATCH --error=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.err

##########################################################
# Extract 3'UTR regions from HAL archive to a FASTA file
# all species
#
# 
# 
#sbatch --array=1-100 extract_train.sh
##########################################################

MAX_LEN=5000000000 #maximum 3'UTR length to extract

TOTAL_RUNS=100 

data_dir=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/

hal_file="$data_dir/zoonomia/241-mammalian-2020v2.hal"
utr_table="$data_dir/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed" 

output_dir="$data_dir/fasta/aligned_3UTR/"

run_idx=${SLURM_ARRAY_TASK_ID} 

while read -r chrom human_UTR_start human_UTR_end UTR_ID score strand transcript_ID canonical HGNC_Symbol UTR_len; do

    length=$(( $MAX_LEN < $UTR_len ? $MAX_LEN : $UTR_len )) #limit length to MAX_LENGTH

    mkdir -p $output_dir/fa/$run_idx
    mkdir -p $output_dir/maf/$run_idx

    maf_file=$output_dir/maf/$run_idx/${UTR_ID}.maf 
    fasta_file=$output_dir/fa/$run_idx/${UTR_ID}.fa 

    echo "extracting alignment for $UTR_ID"

    hal2maf $hal_file  $maf_file --noAncestors --onlyOrthologs  --refGenome Homo_sapiens --refSequence $chrom --start $human_UTR_start --length $length  

    echo "converting to FASTA"

    ./maf_to_fasta.sh ${UTR_ID} $length $strand $maf_file > $fasta_file #take reverse complement for genes on negative strand

    samtools faidx $fasta_file #&& rm $maf_file

done < <(cat  $utr_table|sed -n "${run_idx}~${TOTAL_RUNS}p")
