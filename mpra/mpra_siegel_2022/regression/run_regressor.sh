#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-55 run_regressor.sh
#################################################################

#SBATCH -J mpra_regr
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/MLM/siegel_2022/slurm_logs/%a.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/MLM/siegel_2022/slurm_logs/%a.e

source ~/.bashrc; conda activate mlm

export LD_LIBRARY_PATH=~/miniconda3/lib

c=0

for onlyref in 0 1;do
    for regressor in Ridge; do

    output_dir="/lustre/groups/epigenereg01/workspace/projects/vale/MLM/siegel_2022/predictions/onlyref_$onlyref/${regressor}/"
    
    mkdir -p $output_dir
    
    for response in stability steady_state; do
    
        for cell_type in Jurkat Beas2B; do
    
            for model in 5mers effective_length DNABERT DNABERT-2 NT-MS-v2-500M Species-agnostic Species-aware; do
    
                if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then
    
                    #if ! [ -f "$output_dir/${cell_type}-${response}-${model}.tsv" ]; then
    
                        echo $output_dir/${cell_type}-${response}-${model}.tsv
                        params="--cell_type $cell_type --model $model --output_dir $output_dir \
                    --N_trials 1000 --keep_first --response $response --onlyref $onlyref --N_splits 1000 --N_CVsplits 5 --seed 1  --n_jobs 16 --regressor $regressor"
    
                        python run_regressor.py ${params} > ${output_dir}/${cell_type}-${response}-${model}.log  2>${output_dir}/${cell_type}-${response}-${model}.err 
    
                    #fi
                fi
    
                c=$((c+1))
            done
        done
    done
done
done
        
