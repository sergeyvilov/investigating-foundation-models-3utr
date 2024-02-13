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

dist.init_process_group(backend='nccl', world_size=world_size, rank=rank,)
                       #init_method='file://'+output_dir+'/sharedfile')

print('init complete')
