"""Seeding, device, and distributed-training helpers.

DDP setup follows the standard PyTorch pattern:
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
(torchrun sets RANK / WORLD_SIZE / LOCAL_RANK in the environment)
"""
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int, rank: int = 0) -> None:
    # Offset by rank so DDP workers don't all sample identical augmentations.
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_ddp():
    """Initialize DDP if launched via torchrun (WORLD_SIZE>1 in env). No-op otherwise.

    Returns (rank, local_rank, world_size).
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def reduce_mean(value: float, device) -> float:
    """Average a python scalar across all DDP processes."""
    if not is_dist():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item()) / dist.get_world_size()
