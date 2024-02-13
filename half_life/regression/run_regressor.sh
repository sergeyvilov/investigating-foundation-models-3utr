#!/bin/bash

#################################################################
#
#Run SVR on Siegel MPRA data
#
#
#sbatch --array=0-18 run_regressor.sh
#################################################################

#SBATCH -J half_life
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

c=0

for regressor in Ridge SVR MLP; do

    output_dir="${data_dir}/half_life/agarwal_2022/predictions/${regressor}/"

    mkdir -p $output_dir

        for model in 3K dnabert dnabert2 ntrans-v2-250m dnabert-3utr dnabert2-3utr ntrans-v2-250m-3utr stspace stspace-spaw BCMS-stspace BCMS-stspace-spaw; do
        
                if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

        
                    if [[ ! $model =~ "mers" ]] && [[ ! $model =~ "stspace" ]] && [[ $model != "3K" ]] && [[ $model != "word2vec" ]]; then
                        embeddings="--embeddings $data_dir/human_3utr/embeddings/$model/predictions.pickle"
                    elif [[ $model =~ "stspace-spaw" ]]; then
                        embeddings="--embeddings $data_dir/human_3utr/probs/stspace-spaw/predictions.pickle"
                    elif [[ $model =~ "stspace" ]]; then
                        embeddings="--embeddings $data_dir/human_3utr/probs/stspace/predictions.pickle"
                    fi
                    
                    if [[ $regressor = "MLP" ]]; then
                        n_hpp_trials=150
                    else
                        n_hpp_trials=300
                    fi   

                    output_name=$output_dir/${model}.tsv

                    #if ! [ -f "${output_name}" ]; then

                        echo $output_name

                        params="--model $model $embeddings --regressor $regressor \
                        --n_hpp_trials ${n_hpp_trials} --cv_splits_hpp 5  \
                        --output_name $output_name --n_jobs 10 "

                        python -u run_regressor.py ${params} > ${output_dir}/${model}.log  2>${output_dir}/${model}.err
                    #fi
                fi

                c=$((c+1))
    done
done
