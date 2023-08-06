import os
import torch
from models.vgg import vgg11_bn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

def func(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # create model and move it to GPU with id rank
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    model = vgg11_bn()
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(64, 3, 32, 32).to(device))
    labels = torch.randn(64, 10).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus
    mp.spawn(func, args=(world_size,), nprocs=world_size, join=True)
