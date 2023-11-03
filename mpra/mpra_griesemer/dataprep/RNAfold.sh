#!/bin/bash

#extract  minimal free energy for each seqeunce in FASTA file using RNAfold

RNA_fold_command='/s/project/mll/sergey/effect_prediction/tools/ViennaRNA/bin/RNAfold  --noPS --noLP  '

workdir='/lustre/groups/epigenereg01/workspace/projects/vale/MLM/griesemer/variants/fasta/'

fasta=$1
output_RNAfold=$(basename $fasta .fa)'.free_energy.tsv'


cat $fasta|sed '1d'|sed 's/>.*/ /'|tr -d '\n'|tr ' ' '\n'|$RNA_fold_command \
|sed -n '1~2!p'|sed 's/.* //'|sed 's/[\(\)]//g' > $output_RNAfold
