#!/bin/bash

#################################################################
#
#Generate probabilities and embeddings
#
#################################################################

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

species_list='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_species.txt'

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

logs_dir=$data_dir'human_3utr/probs/'

cd /home/icb/sergey.vilov/workspace/MLM/models/zoonomia/state_space/

for model_name in stspace-3utr-2e stspace-spaw-3utr-2e  stspace-spaw-3utr-DNA stspace-3utr-DNA stspace-3utr-hs; do

    for mask_at_test in true ; do

        checkpoint_dir=$data_dir"models/zoonomia-3utr/$model_name/checkpoints"
    
        if [[ $model == *"DNA"* ]]; then
          fasta=$data_dir'fasta/Homo_sapiens_dna_fwd.fa'
        else
          fasta=$data_dir'fasta/Homo_sapiens_rna.fa'
        fi
            
        if [ "$model_name" = "stspace-3utr-2e" ]; then
          chkpt=48
        elif [ "$model_name" = "stspace-spaw-3utr-2e" ]; then
          chkpt=64
        elif [ "$model_name" = "stspace-spaw-3utr-DNA" ]; then
          chkpt=26
        elif [ "$model_name" = "stspace-3utr-DNA" ]; then
          chkpt=20
        elif [ "$model_name" = "stspace-3utr-hs" ]; then
          chkpt=12
        fi
    
        checkpoint_dir=${checkpoint_dir}/chkpt_${chkpt}

        #output_dir="$logs_dir/$model_name/mask_at_test_${mask_at_test}/"
        output_dir="$logs_dir/$model_name/"

        echo "Output dir: $output_dir"
    
        if [[ ! $model_name =~ 'stspace-spaw' ]]; then
        	is_species_agnostic='--species_agnostic'
        fi
    
        NN_PARAMETERS="--test_dataset $fasta  --species_list ${species_list} \
        --output_dir ${output_dir} $is_species_agnostic \
        --mask_at_test ${mask_at_test} --checkpoint_dir $checkpoint_dir --get_probs --get_embeddings --mask_at_test ${mask_at_test}"
    
        mkdir -p $output_dir

        srun -p gpu_p --qos=gpu_normal -o ${output_dir}/log.o -e ${output_dir}/log.e --nice=10 -J $model_name-3utr -c 4 --mem=64G --gres=gpu:1 --time=1-00:00:00 python -u main_parallel.py ${NN_PARAMETERS} &
    
done

done

