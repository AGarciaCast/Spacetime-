"""Utility functions and layers for architecture building blocks."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define activation functions
ACTS = {
    "tanh": torch.tanh,
    "atan": torch.atan,
    "sigmoid": torch.sigmoid,
    "softplus": F.softplus,
    "relu": F.relu,
    "exp": torch.exp,
    "elu": F.elu,
    "gelu": F.gelu,
    "sin": torch.sin,
    "sinc": lambda z: torch.where(z == 0, torch.ones_like(z), torch.sin(z) / z),
    "linear": lambda z: z,
    "abs_linear": torch.abs,
    "gauss": lambda z: torch.exp(-(z**2)),
    "swish": lambda z: z * torch.sigmoid(z),
    "laplace": lambda z: torch.exp(-torch.abs(z)),
    "gauslace": lambda z: torch.exp(-(z**2)) + torch.exp(-torch.abs(z)),
}


def _resolve_act_name(act):
    if isinstance(act, str):
        parts = act.split("-")
        if len(parts) == 1:
            return act
        if len(parts) == 3 and parts[1] in ACTS:
            return parts[1]
    return None


def _init_weight_tensor(weight, act_name, is_first=False, omega0=30.0):
    """Activation-aware initialization for a raw weight tensor."""
    if act_name == "sin":
        fan_in = weight.shape[1]
        if is_first:
            bound = 1.0 / fan_in
        else:
            bound = (6.0 / fan_in) ** 0.5 / float(omega0)
        nn.init.uniform_(weight, -bound, bound)
    elif act_name in {"relu", "elu", "gelu", "swish", "gauss"}:
        nn.init.kaiming_normal_(weight, nonlinearity="relu")
    elif act_name == "tanh":
        nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain("tanh"))
    else:
        nn.init.xavier_normal_(weight)


def _init_linear_weights(layer, act_name, is_first=False, omega0=30.0):
    if not isinstance(layer, nn.Linear):
        return
    _init_weight_tensor(layer.weight, act_name, is_first=is_first, omega0=omega0)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class AdaptiveActivation(nn.Module):
    """Layer for adaptive activation functions."""

    def __init__(self, act, **kwargs):
        super(AdaptiveActivation, self).__init__(**kwargs)
        parts = act.split("-")
        assert len(parts) == 3, "Activation format should be '(ad)-activation_name-n'"

        self.adapt = "ad" in parts[0]
        self.act = ACTS[parts[1]]
        self.n = float(parts[-1]) if parts[-1].isdigit() else 1.0

        if self.adapt:
            self.a = nn.Parameter(torch.ones(1))  # Trainable weight

    def forward(self, x):
        if self.adapt:
            return self.act(self.n * self.a * x)
        else:
            return self.act(self.n * x)

    def extra_repr(self):
        return f"adapt={self.adapt}, n={self.n}, act={self.act}"


def Activation(act):
    """Parse an activation spec into a callable or AdaptiveActivation module."""
    if callable(act):
        return act
    elif isinstance(act, str):
        parts = act.split("-")
        if len(parts) == 1:
            if act in ACTS.keys():
                return ACTS[act]
            else:
                raise ValueError(f"Unsupported activation: {act}")
        else:
            return AdaptiveActivation(act)
    else:
        raise ValueError("'act' must be either a 'str' or a 'callable'")


def get_activation_fn(name):
    """Normalize activation names or callables into a callable function/module."""
    if callable(name):
        return name
    if isinstance(name, str) and name in ACTS:
        return ACTS[name]
    if isinstance(name, str):
        return Activation(name)
    raise ValueError(f"Unsupported activation: {name}")


def activation_name(act) -> Optional[str]:
    """Public helper for extracting activation name."""
    return _resolve_act_name(act)
