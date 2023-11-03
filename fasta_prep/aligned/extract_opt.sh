#!/bin/bash

#SBATCH -J MSA_coverage
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/aligned/data/logs/%A_%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/aligned/data/logs/%A_%a.e
#SBATCH -c 2
#SBATCH --mem=24G

##########################################################
# Extract 3'UTR regions from HAL archive to a FASTA file
# all species
#
# 
# 
#sbatch --array=0-999%10 extract_train.sh
##########################################################

MAX_LEN=5000 #maximum 3'UTR length to extract

LINES_PER_RUN=19 #number of lines processed in each task

data_dir=/s/project/mll/sergey/effect_prediction/MLM

hal_file="$data_dir/600_way/241-mammalian-2020v2.hal"
utr_table="$data_dir/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed" 

output_dir="$data_dir/aligned/data"

mkdir -p $output_dir/tmp #temporary folder for maf files

run_idx=${SLURM_ARRAY_TASK_ID} #if SLURM_ARRAY_TASK_ID starts by 1 then subtract 1

start_idx=$(($run_idx*$LINES_PER_RUN+1)) #1-based,inclusive
stop_idx=$(($start_idx+$LINES_PER_RUN-1)) #1-based,inclusive

echo "Lines $start_idx-$stop_idx" #lines processed in current run

mkdir -p $output_dir/3_prime_UTR/$run_idx

while read -r chrom human_UTR_start human_UTR_end UTR_ID score strand transcript_ID canonical HGNC_Symbol UTR_len; do

    length=$(( $MAX_LEN < $UTR_len ? $MAX_LEN : $UTR_len )) #limit length to MAX_LENGTH

    maf_file=$output_dir/tmp/${UTR_ID}.maf #temporary maf file

    echo "extracting alignment for $UTR_ID"
    
    fasta_size=$(stat -c %s $output_dir/3_prime_UTR/$run_idx/${UTR_ID}.fa)

    #hal2maf $hal_file  $maf_file --noAncestors --onlyOrthologs  --refGenome Homo_sapiens --refSequence $chrom --start $human_UTR_start --length $length  

    echo "converting to FASTA"

    if [ "$fasta_size" = 0 ] && [ -f "$maf_file" ];then
    ./maf_to_fasta.sh ${UTR_ID} $length $strand $maf_file > $output_dir/3_prime_UTR/$run_idx/${UTR_ID}.fa #take reverse complement for genes on negative strand

    samtools faidx $output_dir/3_prime_UTR/$run_idx/${UTR_ID}.fa && rm $maf_file
    fi
done < <(cat  $utr_table|sed -n "${start_idx},${stop_idx}p")
