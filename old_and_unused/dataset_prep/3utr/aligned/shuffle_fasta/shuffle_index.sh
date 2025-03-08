#!/bin/bash


workdir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/aligned_3UTR/'

seq_per_split=5000

cd $workdir

find ./fa -name '*.fai'|xargs cat|shuf > aligned_seq.shuffled.fai

mkdir -p shuffled_split/fai

cat aligned_seq.fai|awk -v seq_per_split=$seq_per_split 'BEGIN{output_idx=0}{if (NR%seq_per_split==0){output_idx=NR/seq_per_split}{print $0>"shuffled_split/fai/split_"output_idx".fai"}}'
