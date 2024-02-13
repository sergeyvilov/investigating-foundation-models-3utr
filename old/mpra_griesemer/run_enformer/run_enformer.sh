#!/bin/bash

#SBATCH -J enformer
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/griesemer/enformer/logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/griesemer/enformer/logs/%a.e

source ~/.bashrc; conda activate svilov-enformer

task_id=${SLURM_ARRAY_TASK_ID}

start_row=$(($task_id*500))
stop_row=$((($task_id+1)*500))

python -u run_enformer.py $start_row $stop_row