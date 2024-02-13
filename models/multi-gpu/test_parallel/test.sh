#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/icb/sergey.vilov/workspace/MLM/test_parallel/logs/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8500
export WORLD_SIZE=3

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/.bashrc; conda activate ntrans
export LD_LIBRARY_PATH=~/miniconda3/lib

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

srun  python -u parallel.py  > parallel.log 2>&1
