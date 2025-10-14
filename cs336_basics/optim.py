import math
import torch


def get_lr_cosine_schedule(t, lr_max, lr_min, t_warmup, t_cycle):
    assert t_warmup < t_cycle
    assert lr_min < lr_max
    if t < t_warmup:
        return lr_max * t / t_warmup
    if t_cycle <= t:
        return lr_min
    theta = math.pi * (t - t_warmup) / (t_cycle - t_warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(theta))


def apply_gradient_clipping(parameters, max_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm_sq = torch.zeros((), device=grads[0].device)
    for g in grads:
        total_norm_sq += g.data.float().pow(2).sum()
    total_norm = total_norm_sq.sqrt()
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.data.mul_(scale)


class AdamW(torch.optim.AdamW):
    def __init__(
        self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.setdefault("m", torch.zeros_like(p))
                v = state.setdefault("v", torch.zeros_like(p))
                step = state.setdefault("step", 1)
                grad = p.grad.data
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr_t = lr * (1 - beta2 ** step) ** 0.5 / (1 - beta1 ** step)
                p.data.addcdiv_(m, v.sqrt() + eps, value=-lr_t)
                p.data.mul_(1 - lr * weight_decay)
                state["step"] = step + 1
        return loss
