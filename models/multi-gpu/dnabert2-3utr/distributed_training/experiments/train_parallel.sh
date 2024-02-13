#!/bin/bash
#SBATCH --job-name=DNABERT2
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
##SBATCH --constraint=p40&gmem24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH -w supergpu03
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/models/dnabert2-3utr/
#SBATCH --output=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8892
export WORLD_SIZE=2

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

fasta='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/dnabert2-3utr/fasta/chunk_1024_overlap_128.fa'
output_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/dnabert2-3utr/v100_parallel/'

mkdir -p $output_dir
### the command to run
rm -f $output_dir/sharedfile

srun  python -u main_parallel.py --fasta $fasta --output_dir $output_dir --batch_size 256 > $output_dir/log 2>&1
