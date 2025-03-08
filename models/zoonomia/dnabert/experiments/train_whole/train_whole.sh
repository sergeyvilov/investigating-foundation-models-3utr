#!/bin/bash
#SBATCH --job-name=DNABERT
#SBATCH --partition=gpu_p
#SBATCH --time=6-00:00:00
#SBATCH --nice=10000
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --reservation=llm_dna_sequence
#SBATCH --qos gpu_reservation
#SBATCH --chdir=/home/icb/sergey.vilov/workspace/MLM/models/zoonomia/dnabert/
#SBATCH --output=/lustre/groups/epigenereg01/workspace/projects/vale/mlm/slurm_logs/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8500
export WORLD_SIZE=10

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

#fasta='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_mammals.shuffled.fa'
dataset='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/whole_genome/all_species_shuffled.txt'
#dataset='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/whole_genome/test_ds.txt'

output_dir='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/models/zoonomia/dnabert-t/'

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
--save_at -1 8:100000:8 --species_agnostic  --seed 1 --mixed_precision \
--grad_accum_itr 3 --batch_size 64 --steps_per_chkpt 100 --tot_chkpt 10"

srun python -u main_parallel.py  ${NN_PARAMETERS} > ${output_dir}/log 2>&1
