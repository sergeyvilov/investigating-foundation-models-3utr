#!/bin/bash


workdir='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals'

seq_per_split=5000

cd $workdir

shuf 240_mammals.fa.fai > 240_mammals.shuffled.fai

mkdir -p shuffled_split/fai

cat 240_mammals.shuffled.fai|awk -v seq_per_split=$seq_per_split 'BEGIN{output_idx=0}{if (NR%seq_per_split==0){output_idx=NR/seq_per_split}{print $0>"shuffled_split/fai/split_"output_idx".fai"}}'
