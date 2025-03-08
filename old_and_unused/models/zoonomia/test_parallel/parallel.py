import os
import torch
import torch.distributed as dist


assert "WORLD_SIZE" in os.environ
assert 'SLURM_PROCID' in os.environ

world_size = int(os.environ["WORLD_SIZE"])
ngpus_per_node = torch.cuda.device_count()
rank = int(os.environ['SLURM_PROCID'])
gpu = rank % ngpus_per_node

print(f'world size:{world_size}, rank:{rank}, gpus on current node:{ngpus_per_node}, current gpu:{gpu}')

fasta='/lustre/groups/epigenereg01/workspace/projects/vale/mlm/fasta/241_mammals.shuffled.fa'

import datetime

dist.init_process_group(backend='nccl', world_size=world_size, rank=rank,timeout=datetime.timedelta(seconds=120))
                       #init_method='file://'+output_dir+'/sharedfile')


def copy_to_local_scratch(fasta,local_scratch):

    '''
    Copy fasta file to local scratch to ensure minimum delay for data access across all nodes
    '''

    import subprocess
    import os

    md5=subprocess.check_output(f'tail {fasta}|md5sum', shell=True).decode('utf-8').split()[0]

    basename_fa=fasta.split('/')[-1].replace('.fa','')

    local_fasta=f'{local_scratch}/{basename_fa}-{md5}.fa'

    if not os.path.isfile(local_fasta):
        print(f'local dataset {local_fasta} not found, copying from {fasta}')
        os.makedirs(local_scratch, exist_ok=True)
        os.system(f'cp {fasta}.fai {local_fasta}.fai')
        os.system(f'cp {fasta} {local_fasta}')
    else:
        md5_local=subprocess.check_output(f'tail {local_fasta}|md5sum', shell=True).decode('utf-8').split()[0]
        if md5 == md5_local and rank!=0:
            print(f'using {local_fasta} as a valid copy of {fasta}')
        else:
            raise Exception("md5 doesn't match, copying can be in progress...\nexiting...")

    return local_fasta

local_scratch = '/localscratch/sergey.vilov/'

copy_to_local_scratch(fasta,local_scratch)

print('init complete')



while True:
    pass
