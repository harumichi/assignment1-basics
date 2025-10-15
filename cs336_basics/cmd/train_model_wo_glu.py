import time
import os
import logging
import torch

import cs336_basics  # triggers logging configuration in package __init__
from cs336_basics.trainer import train


# Replace SwiGLU with one without GLU
def _SwiGLU_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(self.silu(self.w1(x)))

cs336_basics.nn.SwiGLU.forward = _SwiGLU_forward

logger = logging.getLogger(__name__)

train_path = "data/TinyStoriesV2-GPT4-train.npy"
valid_path = "data/TinyStoriesV2-GPT4-valid.npy"
total_tokens = 327_680_000
batch_size = 32
context_length = 256
max_steps = total_tokens // batch_size // context_length
run_name = 'tinystories/wo_glu'

args = dict(
    train_path=train_path,
    valid_path=valid_path,
    # training loop
    batch_size=batch_size,
    vocab_size=10000,
    context_length=context_length,
    max_steps=max_steps,
    eval_interval=1000,
    # optimization
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    lr_cosine_schedule=None,
    gradient_clipping=None,
    # model parameters
    d_model=512,
    d_ff=2048,  # NOTE: Keep parameter count similar
    num_layers=4,
    num_heads=16,
    rope_theta=10000.0,
    # tensorboard
    tensorboard_dir=f"{os.getcwd()}/tensorboard",
    run_name=run_name,
    # checkpointing
    save_checkpoint_path=None,
    load_checkpoint_path=None,
)
logger.info("Training with args: %s", args)

train(**args)
logger.info("Training finished.")
