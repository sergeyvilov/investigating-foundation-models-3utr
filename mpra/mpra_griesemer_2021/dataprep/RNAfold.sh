#!/bin/bash

#extract  minimal free energy for each seqeunce in FASTA file using RNAfold

RNA_fold_command='/lustre/groups/epigenereg01/workspace/projects/vale/tools/ViennaRNA/bin/RNAfold  --noPS --noLP  '

fasta=$1
output_RNAfold=$2

cat $fasta|sed '1d'|sed 's/>.*/ /'|tr -d '\n'|tr ' ' '\n'|$RNA_fold_command \
|sed -n '1~2!p'|sed 's/.* //'|sed 's/[\(\)]//g' > $output_RNAfold
