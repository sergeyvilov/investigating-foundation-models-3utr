#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-29%10 run_regressor.sh
#################################################################

#SBATCH -J mpra_regr
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/MLM/griesemer/slurm_logs/%a.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/MLM/griesemer/slurm_logs/%a.e

source ~/.bashrc; conda activate mlm

export LD_LIBRARY_PATH=~/miniconda3/lib

regressor='Ridge'

output_dir="/lustre/groups/epigenereg01/workspace/projects/vale/MLM/griesemer/${regressor}_LeaveGroupOut"

c=0

for cell_type in HMEC HEK293FT HEPG2 K562 GM12878 SKNSH; do
#    for model in DNABERT DNABERT-2 NT-MS-v2-500M Species-agnostic; do
     for model in DNABERT DNABERT-2 NT-MS-v2-500M; do   
        if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

            mkdir -p $output_dir

        #    if ! [ -f "$output_dir/${cell_type}-${model}.tsv" ]; then

                #only if output file doesn't exist
                
                echo $output_dir/${cell_type}-${model}.tsv
            
                params="--cell_type $cell_type --model $model --regressor $regressor --output_dir $output_dir \
                --N_trials 2000 --keep_first --N_splits 1000 --N_CVsplits 5 --seed 1 --n_jobs 16"

                python -u run_regressor.py ${params} > ${output_dir}/${cell_type}-${model}.log 2>${output_dir}/${cell_type}-${model}.err 
         #   fi
        fi
        
        c=$((c+1))
        
    done
done
        

        
