#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-41%10 run_svr.sh
#################################################################

#SBATCH -J mpra_svr
#SBATCH -c 8
#SBATCH --mem=25G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-spade

svr_dir='/s/project/mll/sergey/effect_prediction/MLM/griesemer/SVR/skew_pred'

c=0

for cell_type in HMEC HEK293FT HEPG2 K562 GM12878 SKNSH; do
    
    for model in enformer_all_targets; do
#    for model in MLM 4mers 5mers word2vec griesemer enformer_all_targets enformer_summary; do
    
        if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then
        
            output_dir="$svr_dir/$cell_type/$model"

            params="--cell_type $cell_type --model $model --output_dir $output_dir \
            --N_trials 1000 --n_jobs 8 --N_splits 1000 --N_CVsplits 5 --seed 1"

            mkdir -p $output_dir

            python -u run_svr.py ${params} > ${output_dir}/log 2>${output_dir}/err 
        fi
        
        c=$((c+1))
        
    done
done
        

        
