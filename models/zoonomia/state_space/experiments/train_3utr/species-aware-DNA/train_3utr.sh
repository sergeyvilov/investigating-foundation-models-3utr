#!/bin/bash
#SBATCH --job-name=STSPACE
#SBATCH --partition=gpu_p
#SBATCH --time=4-00:00:00
#SBATCH --nice=10000
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --qos gpu_long
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/models/zoonomia/state_space/
#SBATCH --output=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8503
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source /home/icb/sergey.vilov/.bashrc; conda activate ntrans

export LD_LIBRARY_PATH=/home/icb/sergey.vilov/miniconda3/lib

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL

dataset='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_mammals_DNA.shuffled.fa'

output_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/models/zoonomia-3utr/stspace-spaw-3utr-DNA/'

species_list='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_species.txt'

mkdir -p $output_dir

#model_checkpoint='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/models/zoonomia/dnabert2-t/checkpoints/chkpt_2/'
#model_checkpoint=-1

if [ ! -z "${model_checkpoint}" ]; then
    if [ "${model_checkpoint}" = "-1" ]; then
        #get last checkpoint
        checkpoint_dir="$output_dir/checkpoints"
        last_epoch=$(ls $checkpoint_dir|grep "chkpt_"|cut -d"_" -f2|sort -n|tail -n1)
        if [ ! -z "${last_epoch}" ]; then
            checkpoint="--checkpoint_dir $checkpoint_dir/chkpt_${last_epoch}"
        fi
    else
        checkpoint="--checkpoint_dir ${model_checkpoint}"
    fi
fi

NN_PARAMETERS="${COMMON_NN_PARAMETERS}  \
--train_dataset $dataset  --output_dir ${output_dir} ${checkpoint} \
--save_at -1 4:100000:4  --seed 1  \
--grad_accum_itr 5 --batch_size 256 --steps_per_chkpt 500 --tot_chkpt 10000 --species_list ${species_list}"

srun python -u main_parallel.py  ${NN_PARAMETERS} > ${output_dir}/log 2>&1
