#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-120 run_regressor.sh
#################################################################

#SBATCH -J gries_regr
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
##SBATCH --time=2-00:00:00
#SBATCH --nice=10000
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.e

data_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/mpra/griesemer_2021/'

mpra_tsv="${data_dir}/mpra_rna.tsv"

source ~/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=~/miniconda3/lib

c=0

for regressor in MLP; do

    output_dir="${data_dir}/predictions/${regressor}/"

    mkdir -p $output_dir

    for cell_type in HMEC HEK293FT HEPG2 K562 GM12878 SKNSH; do

        for model in dnabert dnabert2 ntrans-v2-500m ntrans-v2-250m dnabert-3utr dnabert2-3utr ntrans-v2-250m-3utr griesemer stspace stspace-spaw; do

            if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

                if [[ ! $model =~ "griesemer" ]]; then
                    embeddings="--embeddings $data_dir/embeddings/$model/predictions.pickle"
                fi

                if [[ $regressor = "MLP" ]]; then
                    n_hpp_trials=150
                else
                    n_hpp_trials=300
                fi

                output_name=$output_dir/${cell_type}-${model}.tsv

                #if ! [ -f "${output_name}" ]; then

                        echo $output_name

                        params="--mpra_tsv $mpra_tsv --cell_type $cell_type --model $model $embeddings \
                        --regressor $regressor
                        --n_hpp_trials ${n_hpp_trials} --cv_splits_hpp 5 \
                        --output_name $output_name --seed 1  --n_jobs 10 "

                        python -u run_regressor.py ${params} > ${output_dir}/${cell_type}-${model}.log  2>${output_dir}/${cell_type}-${model}.err

                #fi
            fi
            c=$((c+1))
        done
    done
done
