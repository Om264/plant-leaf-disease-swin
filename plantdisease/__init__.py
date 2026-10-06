"""Plant leaf disease classification with timm Swin Transformer.

Training recipe follows:
  - Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows", ICCV 2021. https://arxiv.org/abs/2103.14030
  - The official Swin-Transformer config defaults (AdamW, cosine+warmup,
    gradient clipping, per-size stochastic depth, Mixup=0.8/CutMix=1.0):
    https://github.com/microsoft/Swin-Transformer
  - timm's "sw" (Swin) training recipe tag: AdamW + gradient clipping + EMA,
    cosine with warmup. https://huggingface.co/docs/timm/hparams
"""
__version__ = "1.0.0"
