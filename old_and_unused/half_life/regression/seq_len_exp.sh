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
#SBATCH --time=3-00:00:00
#SBATCH --nice=10000
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%A_%a.e

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

c=0

model=ntrans-v2-100m-3utr-2e

for regressor in Ridge SVR; do

    output_dir="${data_dir}/half_life/agarwal_2022/seqlen-exp/predictions/${regressor}/"

    mkdir -p $output_dir

        for max_seq_len in 512 1024 2048 4096; do

            if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

                embeddings="--embeddings $data_dir/half_life/agarwal_2022/seqlen-exp/embeddings/$model/$max_seq_len/predictions.pickle"
    
                    if [[ $regressor = "MLP" ]]; then
                        n_hpp_trials=150
                    else
                        n_hpp_trials=300
                    fi

                    output_name=$output_dir/${model}-${max_seq_len}.tsv

                    if ! [ -f "${output_name}" ]; then

                        echo $output_name

                        params="$embeddings --regressor $regressor \
                        --n_hpp_trials ${n_hpp_trials} --cv_splits_hpp 5 --n_jobs 10 "

                        python -u run_regressor.py ${params} --model $model --output_name $output_name > ${output_dir}/${model}-${max_seq_len}.log  2>${output_dir}/${model}-${max_seq_len}.err

                    fi
                fi

                c=$((c+1))
    done
done
