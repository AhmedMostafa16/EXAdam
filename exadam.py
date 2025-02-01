"""
MIT License

Copyright (c) 2024 Ahmed M. Adly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Any, Dict, Iterable, Tuple, Union
import numpy as np
import torch

Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class EXAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,  # Model parameters
        lr: Union[float, torch.Tensor] = 0.001,  # Learning rate (default: 0.001)
        betas: Tuple[float, float] = (
            0.9,
            0.999,
        ),  # Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: float = 1e-8,  # Epsilon value added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: float = 0.0,  # Weight decay (L2 penalty) (default: 0.0)
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(betas[0]))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(betas[1]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Pre-compute sqrt(2) for efficiency
        self.sqrt_2 = np.sqrt(2.0)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr: Union[float, torch.Tensor] = group["lr"]
            beta1: float
            beta2: float
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "EXAdam does not support sparse gradients, please consider Sparsexadam instead"
                    )

                grad: torch.Tensor = p.grad.data

                state = self.state[p]

                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=lr * -weight_decay)

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]

                state["step"] += 1

                step: int = state["step"]

                beta1_t: float = beta1**step
                beta2_t: float = beta2**step

                bias_correction1: float = 1 - beta1_t
                bias_correction2: float = 1 - beta2_t

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(other=grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(tensor1=grad, tensor2=grad, value=1 - beta2)

                # Compute the new debiasing terms
                d1: torch.Tensor = 1 + (v.div(v + eps)) * beta2_t
                d2: torch.Tensor = 1 + (m.pow(2).div(m.pow(2) + eps)) * beta1_t

                m_tilde: torch.Tensor = m.div(bias_correction1) * d1
                v_tilde: torch.Tensor = v.div(bias_correction2) * d2

                # Bias-corrected gradient
                g_tilde = grad.div(bias_correction1) * d1

                # Compute the step size
                step_size = lr * np.log(np.sqrt(step + 1) * self.sqrt_2)

                # Update the parameters
                theta: torch.Tensor = (
                    -step_size * (m_tilde + g_tilde) / (v_tilde.sqrt() + eps)
                )

                p.data.add_(theta)

        return loss
