"""Cosine learning-rate schedule with linear warmup.

Matches the Swin-Transformer paper's fine-tuning schedule (AdamW + cosine
decay + linear warmup), ICCV 2021, Sec. 4.1 / Appendix A1:
https://arxiv.org/abs/2103.14030
"""
import math


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                warmup_lr_init_ratio=0.01, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        self.total_steps = max(self.warmup_steps + 1, total_epochs * steps_per_epoch)
        self.warmup_lr_init_ratio = warmup_lr_init_ratio
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0

    def step(self):
        self.step_count += 1
        t = self.step_count
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if t <= self.warmup_steps:
                frac = t / self.warmup_steps
                lr = base_lr * (self.warmup_lr_init_ratio + (1 - self.warmup_lr_init_ratio) * frac)
            else:
                progress = (t - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                cos = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cos)
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]
