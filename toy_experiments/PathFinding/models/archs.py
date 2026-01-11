"""Neural architectures for toy path-finding experiments."""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from toy_experiments.PathFinding.models.archs_utils import (
    Activation,
    _init_linear_weights,
    _init_weight_tensor,
    _resolve_act_name,
)


def get_backbone(
    backbone_type: str,
) -> nn.Module:
    """Factory function to get backbone architecture."""
    if backbone_type == "mlp":
        return MLP
    elif backbone_type == "pirate_net":
        return PirateNet
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")


class FourierEmbs(nn.Module):
    """Optional Fourier feature embedding."""

    def __init__(self, embed_scale: float, embed_dim: int) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even.")
        self.embed_scale = float(embed_scale)
        self.embed_dim = int(embed_dim)
        self.kernel = None

    def forward(self, x):
        if self.kernel is None:
            in_dim = x.shape[-1]
            kernel = torch.empty(
                in_dim, self.embed_dim // 2, device=x.device, dtype=x.dtype
            )
            nn.init.normal_(kernel, mean=0.0, std=self.embed_scale)
            self.kernel = nn.Parameter(kernel)
        y = torch.matmul(x, self.kernel)
        return torch.cat([torch.cos(y), torch.sin(y)], dim=-1)


class ReparamDense(nn.Module):
    """Dense layer with optional weight factorization reparameterization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        reparam: Union[None, Dict] = None,
        bias: bool = True,
        activation: Optional[str] = None,
        is_first: bool = False,
        omega0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.reparam = reparam
        self.activation = _resolve_act_name(activation)
        self.is_first = is_first
        self.omega0 = float(omega0)
        self.linear = (
            None
            if reparam
            else nn.Linear(self.in_features, self.out_features, bias=bias)
        )
        self.g = None
        self.v = None
        self.bias = None
        if reparam:
            if self.reparam.get("type") != "weight_fact":
                raise ValueError(
                    f"Unsupported reparam type: {self.reparam.get('type')}"
                )
            w = torch.empty(self.in_features, self.out_features)
            _init_weight_tensor(
                w,
                self.activation,
                is_first=self.is_first,
                omega0=self.omega0,
            )
            g = torch.empty(self.out_features)
            nn.init.normal_(
                g, mean=float(self.reparam["mean"]), std=float(self.reparam["stddev"])
            )
            g = torch.exp(g)
            v = w / g
            self.g = nn.Parameter(g)
            self.v = nn.Parameter(v)
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            _init_linear_weights(
                self.linear, self.activation, is_first=self.is_first, omega0=self.omega0
            )

    def forward(self, x):
        if self.reparam is None:
            return self.linear(x)
        weight = self.v * self.g
        y = torch.matmul(x, weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class MLP(nn.Module):
    """Simple MLP builder with adaptive activations and output activation."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        out_dim=1,
        activation="ad-gauss-1",
        out_activation="linear",
        fourier_emb: Union[None, Dict] = None,
        reparam: Union[None, Dict] = None,
    ):
        super(MLP, self).__init__()

        self.embedding = FourierEmbs(**fourier_emb) if fourier_emb else None

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        assert (
            isinstance(hidden_dim, (list, tuple)) and len(hidden_dim) == num_layers
        ), "num_layers must match the length of hidden_dim"

        self.layers = nn.ModuleList()
        # check if Activation(activation) is a nn.Module
        if isinstance(Activation(activation), nn.Module):
            self.activations = nn.ModuleList()
        else:
            self.activations = []

        # Input layer
        if self.embedding is not None:
            input_dim = int(self.embedding.embed_dim)
        input_linear = ReparamDense(
            input_dim,
            hidden_dim[0],
            reparam=reparam,
            activation=activation,
            is_first=True,
        )
        self.layers.append(input_linear)
        self.activations.append(Activation(activation))
        # Hidden layers
        for i in range(1, num_layers):
            linear = ReparamDense(
                hidden_dim[i - 1],
                hidden_dim[i],
                reparam=reparam,
                activation=activation,
            )
            self.layers.append(linear)
            self.activations.append(Activation(activation))

        # Output layer
        out_act_name = _resolve_act_name(out_activation)
        linear = ReparamDense(
            hidden_dim[-1],
            out_dim,
            reparam=reparam,
            activation=out_act_name,
        )
        self.layers.append(linear)

        self.out_activation = Activation(out_activation)

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activations[i](layer(x))
        x = self.out_activation(self.layers[-1](x))
        return x


class PIModifiedBottleneck(nn.Module):
    """Modified PI bottleneck with gated mixing and skip connection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str,
        nonlinearity: float,
        reparam: Union[None, Dict],
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation_name = activation
        self.act1 = Activation(activation)
        self.act2 = Activation(activation)
        self.act3 = Activation(activation)
        self.reparam = reparam
        self.alpha = nn.Parameter(torch.tensor([float(nonlinearity)]))
        self.dense1 = ReparamDense(
            self.input_dim, self.hidden_dim, reparam=self.reparam, activation=activation
        )
        self.dense2 = ReparamDense(
            self.hidden_dim,
            self.hidden_dim,
            reparam=self.reparam,
            activation=activation,
        )
        self.dense3 = ReparamDense(
            self.hidden_dim,
            self.input_dim,
            reparam=self.reparam,
            activation=self.activation_name,
        )

    def forward(self, x, u, v):
        identity = x

        x = self.dense1(x)
        x = self.act1(x)
        x = x * u + (1 - x) * v

        x = self.dense2(x)
        x = self.act2(x)
        x = x * u + (1 - x) * v

        x = self.dense3(x)
        x = self.act3(x)

        x = self.alpha * x + (1 - self.alpha) * identity
        return x


class PirateNet(nn.Module):
    """PirateNet architecture with PI-style bottlenecks and embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        out_dim: int = 1,
        activation: str = "tanh",
        out_activation: str = "linear",
        nonlinearity: float = 0.0,
        fourier_emb: Union[None, Dict] = None,
        reparam: Union[None, Dict] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.act_u = Activation(activation)
        self.act_v = Activation(activation)
        self.reparam = reparam
        self.embedding = FourierEmbs(**fourier_emb) if fourier_emb else None
        embed_dim = self._embedding_dim()
        self.u_proj = ReparamDense(
            embed_dim,
            self.hidden_dim,
            reparam=self.reparam,
            activation=activation,
            is_first=True,
        )
        self.v_proj = ReparamDense(
            embed_dim,
            self.hidden_dim,
            reparam=self.reparam,
            activation=activation,
            is_first=True,
        )
        self.blocks = nn.ModuleList(
            [
                PIModifiedBottleneck(
                    input_dim=embed_dim,
                    hidden_dim=self.hidden_dim,
                    activation=activation,
                    nonlinearity=nonlinearity,
                    reparam=self.reparam,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_activation = Activation(out_activation)
        out_act_name = _resolve_act_name(out_activation)
        self.out_proj = ReparamDense(
            embed_dim, self.out_dim, reparam=self.reparam, activation=out_act_name
        )

    def _embedding_dim(self):
        dim = self.input_dim
        if self.embedding is not None:
            dim = int(self.embedding.embed_dim)
        return dim

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)

        u = self.u_proj(x)
        u = self.act_u(u)

        v = self.v_proj(x)
        v = self.act_v(v)

        for block in self.blocks:
            x = block(x, u, v)

        x = self.out_activation(self.out_proj(x))

        return x
