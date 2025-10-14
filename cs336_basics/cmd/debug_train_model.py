import time
import logging

import cs336_basics  # triggers logging configuration in package __init__
from cs336_basics.trainer import train

logger = logging.getLogger(__name__)

train_path = "data/dummy_train.npy"
valid_path = "data/dummy_valid.npy"

begin_time = time.time()
train(
    train_path=train_path,
    valid_path=valid_path,
    # training loop
    batch_size=16,
    vocab_size=10000,
    context_length=128,
    max_steps=500,
    eval_interval=100,
    # optimization
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    lr_cosine_schedule=None,
    gradient_clipping=None,
    # model parameters
    d_model=128,
    d_ff=512,
    num_layers=2,
    num_heads=2,
    rope_theta=10000.0,
    # checkpointing
    save_checkpoint_path=None,
    load_checkpoint_path=None,
    wandb_args={"name": "debug-train-model"},
)
end_time = time.time()

logger.info("Training elapsed time: %.2f s", end_time - begin_time)
