import os
import torch.distributed as dist

dist.init_process_group("nccl")
pg = dist.group.WORLD
print("rank", dist.get_rank(), "TORCH_DISTRIBUTED_DEBUG=", os.getenv("TORCH_DISTRIBUTED_DEBUG"))
print("pg type:", type(pg), pg, flush=True)
dist.destroy_process_group()
