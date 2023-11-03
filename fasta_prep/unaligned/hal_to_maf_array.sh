#!/bin/bash

#################################################################
#
#Extract maf files for protein-coding genes
#
#TO RUN: sbatch --array=1-1000%10 hal_to_maf_array.sh
#################################################################


#SBATCH -J MSA_coverage
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -o /s/project/mll/sergey/MLM/exp/600_way_stop_codon/logs/%A_%a.o
#SBATCH -e /s/project/mll/sergey/MLM/exp/600_way_stop_codon/logs/%A_%a.e

#N_genes=2000
#min_gene_length=1000
max_files_per_subdir=500

#source ~/.bashrc; conda activate bio

workdir='/s/project/mll/sergey/MLM/exp/600_way_stop_codon/maf/'

msa_hal='/s/project/mll/sergey/MLM/600_way/241-mammalian-2020v2.hal'

hal2maf='/s/project/mll/sergey/MLM/tools/hal/bin/hal2maf'

segment_list='/s/project/mll/sergey/MLM/UTR_coords/GRCh38_3_prime_UTR_clean.bed'

max_task_id=$(cat $segment_list|wc -l )

for multiplier in {0..20}; do

  row_num=$(($multiplier*1000+${SLURM_ARRAY_TASK_ID}))

  if [ $row_num -gt $max_task_id ];then
    exit 0
  fi

  segment=$(head -n $row_num $segment_list | tail -1)

  read chrom start end UTR_ID _ strand _ <<< $segment

  #transcript_ID=$(echo $UTR_ID|cut -d"." -f1)

  if [ $strand == '+' ];then
    start=$((start-3))
  else
    start=$end
  fi

  length=3

  #output_dir=$workdir/alignments
  output_dir=$workdir/$((row_num/max_files_per_subdir))

  mkdir -p $output_dir
  #mkdir -p $output_dir/a2m

  echo $multiplier $row_num
  echo $segment
  echo $hal2maf $msa_hal stdout --refGenome Homo_sapiens --refSequence $chrom --noAncestors --onlyOrthologs --start $start --length $length

  $hal2maf $msa_hal stdout --refGenome Homo_sapiens --refSequence $chrom --noAncestors --onlyOrthologs --start $start --length $length \
  |grep -v '#' > $output_dir/${UTR_ID}.maf

  #python maf_to_a2m.py $output_dir/maf/${transcript_ID}.maf > $output_dir/a2m/${transcript_ID}.a2m

done
