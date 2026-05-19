"""PyTorch Muon optimizer.

Port of the original Muon (Momentum + Newton-Schulz orthogonalisation) optimizer.
Reference: https://github.com/KellerJordan/Muon

Muon is the default optimizer for ELF-B training (config.optimizer = "muon").
It applies orthogonalised gradient updates via Newton-Schulz iterations,
which empirically outperforms AdamW for transformer training.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute G / ||G||_op (approximate matrix square-root inverse).

    Produces an orthogonal-ish matrix of the same shape as G.
    Works for matrices of any size by computing the SVD approximation iteratively.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype not in (torch.float32, torch.bfloat16) else G.clone()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalise so largest singular value starts near 1
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)      # quintic: a*I + b*A + c*A^2
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon — Momentum + Nesterov-like update with Newton-Schulz orthogonalisation.

    Only 2-D (weight matrix) parameters receive the orthogonalised update.
    1-D parameters (biases, norms) fall back to plain SGD with momentum.

    Args:
        params:       iterable of parameters or param groups
        lr:           learning rate (default 1e-3)
        momentum:     SGD momentum (default 0.95)
        nesterov:     whether to use Nesterov momentum (default True)
        ns_steps:     number of Newton-Schulz iterations (default 6)
        weight_decay: weight decay (L2 penalty) applied to all params (default 0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Weight decay
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                if update.ndim >= 2:
                    # Apply Newton-Schulz orthogonalisation for matrix weights
                    update_orth = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Scale to match the RMS of the raw gradient
                    scale = max(1, update.size(-2) / update.size(-1)) ** 0.5
                    p.add_(update_orth * scale, alpha=-lr)
                else:
                    # Scalars / biases: plain momentum SGD
                    p.add_(update, alpha=-lr)

        return loss
