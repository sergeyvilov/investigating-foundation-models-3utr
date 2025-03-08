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

for regressor in Ridge SVR; do

    output_dir="${data_dir}/half_life/agarwal_2022/predictions/${regressor}/"

    mkdir -p $output_dir


      for model in 3K BC3MS \
                  dnabert dnabert-3utr-2e \
                  dnabert2 dnabert2-zoo dnabert2-3utr-2e \
                  ntrans-v2-100m ntrans-v2-100m-3utr-2e \
                  stspace-3utr-2e stspace-spaw-3utr-2e \
                  stspace-spaw-3utr-DNA stspace-3utr-DNA stspace-3utr-hs; do

                if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

                    if [[ $model != "3K" ]]  && [[ $model != "BC3MS" ]];then
                      if [[ ! $model =~ "stspace" ]]; then
                          embeddings="--embeddings $data_dir/human_3utr/embeddings/$model/predictions.pickle"
                      else
                          embeddings="--embeddings $data_dir/human_3utr/probs/$model/predictions.pickle"
                      fi
                    fi

                    if [[ $regressor = "MLP" ]]; then
                        n_hpp_trials=150
                    else
                        n_hpp_trials=300
                    fi

                    output_name=$output_dir/${model}.tsv

                    if ! [ -f "${output_name}" ]; then

                        echo $output_name

                        params="$embeddings --regressor $regressor \
                        --n_hpp_trials ${n_hpp_trials} --cv_splits_hpp 5 --n_jobs 10 "

                        python -u run_regressor.py ${params} --model $model --output_name $output_name > ${output_dir}/${model}.log  2>${output_dir}/${model}.err

                        if [[ $model =~ "dnabert2-3utr-2e" ]] || [[ $model =~ "ntrans-v2-100m-3utr-2e" ]]; then

                            model=BC3MS-${model}

                            output_name=$output_dir/${model}.tsv

                            python -u run_regressor.py ${params} --model ${model} --output_name $output_name > ${output_dir}/${model}.log  2>${output_dir}/${model}.err
                        fi
                    fi
                fi

                c=$((c+1))
    done
done
