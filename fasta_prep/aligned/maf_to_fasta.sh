#!/bin/bash

UTR_ID=$1
length=$2
strand=$3

maf=$4

LINE_WIDTH=80

while read line; do

if [[ $line == ">"* ]] ; 
then 
    species_name=$(echo ${line:1}|sed 's/Homo_sapiens.*/Homo_sapiens/') #remove contig name that exists for Homo_sapiens only
    line=">${UTR_ID}:${species_name}:${length}" #use human UTR length for all species
    echo $line
else

    if [[ $strand == '-' ]];
    then
        #take reverse complemented
        line=$(echo $line|tr 'ACTGactg-' 'TGACtgac-'|rev)
    fi
    
    echo $line|fold -w $LINE_WIDTH
    
fi

done < <(python /data/ouga/home/ag_gagneur/l_vilov/workspace/species-aware-DNA-LM/dataprep/aligned/fasta/utils/maf_to_a2m.py $maf)


