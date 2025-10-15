import os
from typing import BinaryIO, IO
import logging
from datetime import datetime

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.nn import Transformer, cross_entropy
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, apply_gradient_clipping


logger = logging.getLogger(__name__)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    # 2D indices: (B, T)
    idx = starts[:, None] + np.arange(context_length)
    inputs = dataset[idx]
    targets = dataset[idx + 1]
    inputs = torch.from_numpy(inputs).to(device, dtype=torch.long)
    targets = torch.from_numpy(targets).to(device, dtype=torch.long)
    return inputs, targets


def get_all_batches(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str,
):
    num_batches = (len(dataset) - 1) // (batch_size * context_length)
    if num_batches == 0:
        return

    for i in range(num_batches):
        b = batch_size * context_length * i
        e = batch_size * context_length * (i + 1)
        input_batch = dataset[b:e].reshape(batch_size, context_length)
        target_batch = dataset[b + 1 : e + 1].reshape(batch_size, context_length)
        yield (
            torch.tensor(input_batch, dtype=torch.long, device=device),
            torch.tensor(target_batch, dtype=torch.long, device=device),
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def validate(
    *,
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        for input_batch, target_batch in get_all_batches(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        ):
            logits = model(input_batch)
            loss = cross_entropy(logits, target_batch, reduction='sum')
            total_loss += loss.item()
            total_tokens += input_batch.numel()
        avg_loss = total_loss / total_tokens
        ppl = np.exp(avg_loss)
        return avg_loss, ppl


def train(
    *,
    train_path: str,
    valid_path: str,
    # training loop
    batch_size: int,
    vocab_size: int,
    context_length: int,
    max_steps: int,
    eval_interval: int,
    # optimization
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    lr_cosine_schedule: dict | None,
    gradient_clipping: dict | None,
    # model parameter
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float,
    # tensorboard
    tensorboard_dir: str,
    run_name: str | None = None,
    # checkpointing
    save_checkpoint_path: str | os.PathLike | BinaryIO | IO[bytes] | None = None,
    load_checkpoint_path: str | os.PathLike | BinaryIO | IO[bytes] | None = None,
):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d%H%M%S")
    logger.info(f"run_name: {run_name}")
    writer = SummaryWriter(os.path.join(tensorboard_dir, run_name))

    train_dataset = np.load(train_path, mmap_mode="r")
    valid_dataset = np.load(valid_path, mmap_mode="r")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"device: {device}")
    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        rope_theta=rope_theta,
    ).to(device)
    logger.info("model: %s", model)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )
    if lr_cosine_schedule is not None:
        lr_scheduler = lambda lr, step: get_lr_cosine_schedule(
            t=step,
            lr_max=lr,
            lr_min=lr_cosine_schedule.get("lr_min", 0.0),
            t_warmup=lr_cosine_schedule["t_warmup"],
            t_cycle=lr_cosine_schedule["t_cycle"],
        )
    else:
        lr_scheduler = lambda lr, step: lr

    initial_step = 0
    if load_checkpoint_path is not None:
        initial_step = load_checkpoint(
            load_checkpoint_path, model=model, optimizer=optimizer
        )

    logger.info(f"training start from step {initial_step} to {max_steps}")
    for i in range(max_steps - initial_step):
        step = initial_step + i + 1

        new_lr = lr_scheduler(lr, step)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        model.train()
        input_batch, target_batch = get_batch(
            dataset=train_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        optimizer.zero_grad()
        logits = model(input_batch)
        loss = cross_entropy(logits, target_batch)
        loss.backward()
        if gradient_clipping is not None:
            apply_gradient_clipping(model.parameters(), gradient_clipping["max_l2_norm"])
        optimizer.step()

        if step % eval_interval == 0:
            valid_loss, valid_ppl = validate(
                model=model,
                dataset=valid_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logger.info(f"steps: {step}, tokens: {step * batch_size * context_length}, valid loss: {valid_loss:.4f}")
            writer.add_scalar("valid/loss", valid_loss, step * batch_size * context_length)
            writer.add_scalar("valid/ppl", valid_ppl, step * batch_size * context_length)
            if save_checkpoint_path is not None:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step,
                    out=save_checkpoint_path,
                )

    writer.flush()
    writer.close()
