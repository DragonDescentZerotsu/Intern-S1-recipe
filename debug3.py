# /tmp/rs_check_fixed.py
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

dist.init_process_group("nccl")
rank = dist.get_rank()
ws = dist.get_world_size()

def t(name, pg):
    out = torch.zeros(8, device="cuda")
    inp = torch.ones(8 * ws, device="cuda")  # ✅ 必须是 output * world_size
    try:
        w = pg.reduce_scatter_tensor_coalesced([out], [inp], dist.ReduceScatterOptions())
        w.wait()
        print(f"rank{rank} {name}: OK out_sum={out.sum().item()}", flush=True)
    except Exception as e:
        print(f"rank{rank} {name}: FAIL -> {e}", flush=True)

t("WORLD", dist.group.WORLD)
pg = dist.new_group(ranks=list(range(ws)), backend="nccl")
dist.barrier()
t("NEW_GROUP", pg)
dist.destroy_process_group()
