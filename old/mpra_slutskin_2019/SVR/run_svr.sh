#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-5%10 run_svr.sh
#################################################################

#SBATCH -J mpra_svr
#SBATCH -c 8
#SBATCH --mem=25G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-mlm

svr_dir='/s/project/mll/sergey/effect_prediction/MLM/slutskin_2019/SVR/'

c=0

for model in MLM 4mers 5mers 6mers word2vec; do
    
        if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then
        
            output_dir="$svr_dir/$model"

            params="--model $model --output_dir $output_dir \
            --N_trials 1000 --n_jobs 8"

            mkdir -p $output_dir

            python run_svr.py ${params} > ${output_dir}/log 2>${output_dir}/err 
        fi
        
        c=$((c+1))
        
    done
done
        

        
