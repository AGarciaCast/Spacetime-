from typing import Dict, Union, Optional

import torch
from torch import nn


# Import modules
from toy_experiments.PathFinding.models.archs import get_backbone


class NeuralEikonalSolver(nn.Module):
    """Base  Neural Eikonal Solver."""

    def __init__(
        self,
        dim_signal: int,
        backbone_type: str,
        hidden_dim: int,
        num_layers: int,
        activation="ad-gauss-1",
        fourier_emb: Union[None, Dict] = None,
        reparam: Union[None, Dict] = None,
        xmin=None,
        xmax=None,
        factored: bool = True,
        normalize_domain: bool = True,
    ):
        super(NeuralEikonalSolver, self).__init__()

        self.backbone = get_backbone(backbone_type)(
            input_dim=dim_signal + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            fourier_emb=fourier_emb,
            reparam=reparam,
        )

        self.factored = factored
        self.normalize_domain = normalize_domain

        xmin = xmin if xmin is not None else [0.0] * (dim_signal + 1)
        xmax = xmax if xmax is not None else [1.0] * (dim_signal + 1)
        self.register_buffer("xmin", torch.tensor(xmin, dtype=torch.float32))
        self.register_buffer("xmax", torch.tensor(xmax, dtype=torch.float32))

        self.act = nn.GELU()

    def forward(self, inputs):
        """Computes traveltimes.

        Args:
            inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).

        Returns:
        out (torch.Tensor): output.
            Shape (batch_size, 1).
        """

        if self.factored:
            dist = self.ambient_distance(inputs)

        if self.normalize_domain:
            # Normalize inputs to [-1, 1]
            inputs = 2.0 * (inputs - self.xmin) / (self.xmax - self.xmin) - 1.0

        xs, xr = inputs[:, 0, :], inputs[:, 1, :]

        out_1 = self.backbone(torch.cat([xs, xr], dim=-1))
        out_2 = self.backbone(torch.cat([xr, xs], dim=-1))
        out = 0.5 * (out_1 + out_2)
        out = self.act(out)

        if self.factored:
            # The codomain (range) of GELU is approximately [-0.170, +\infty)
            # Also the intrinsic (geodesic) distance on an embedded manifold can never
            # be smaller than the chordal (Euclidean) distance in the ambient space.
            out = dist * (1 + out)
        else:
            # pass through relu to ensure positiveness
            out = torch.relu(out)

        return out

    def times(self, inputs):
        """Computes traveltimes.

        Args:
           inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).

        Returns:
        out (torch.Tensor): output.
            Shape (batch_size,).
        """
        outputs = self.forward(inputs)

        return outputs.squeeze(-1)

    def metric_tensor(self, inputs):
        """Metric tensor at input points.

        Args:
           inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).
        Returns:
        out (torch.Tensor): Metric tensor at input points.
            Shape (batch_size, 2, dim_signal+1, dim_signal+1).
        """
        raise NotImplementedError("metric tensor method must be implemented.")

    def basis_vectors(self, inputs):
        """Basis vectors of the tangent space at input points.

        Args:
           inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).
        Returns:
        out (torch.Tensor): Basis vectors at input points.
            Shape (batch_size, 2, dim_signal+1).
        """
        raise NotImplementedError("basis vectors method must be implemented.")

    def inverse_metric_tensor(self, inputs):
        """Inverse of the metric tensor at input points.

        Args:
           inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).
        Returns:
        out (torch.Tensor): Inverse of the metric tensor at input points.
            Shape (batch_size, 2, dim_signal+1, dim_signal+1).
        """
        g = self.metric_tensor(inputs)
        g_inv = torch.linalg.inv(g)
        return g_inv

    def ambient_distance(self, inputs):
        """Chordal (ambient) distance.

        Args:
            inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).
        Returns:
            dist (torch.Tensor): The chordal distance between input points.
            Shape (batch_size, 1).
        """
        dist = torch.norm(inputs[:, 0, :] - inputs[:, 1, :], dim=-1, keepdim=True)

        return dist

    def project(self, inputs):
        """Project to manifold

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, 2, dim_signal+1).
        Returns:
            projected_inputs (torch.Tensor): The projected input points. Shape (batch_size, 2, dim_signal+1).

        """
        return torch.clamp(inputs, min=self.xmin, max=self.xmax)

    def times_and_gradients(self, inputs, reuse_grad=False):
        """Computes traveltimes, and gradient w.r.t. 'xs' and 'xr' given the time output.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, 2, dim_signal+1).
            reuse_grad (bool): if true then create_graph=True, retain_graph=True
        Returns:
            times (torch.Tensor): The traveltimes. Shape (batch_size,).
            gradients (torch.Tensor): The gradients of traveltimes w.r.t. inputs. Shape (batch_size, 2, dim_signal+1).

        """

        inputs.requires_grad_()

        times = self.forward(inputs)

        euc_gradients = torch.autograd.grad(
            outputs=times,
            inputs=inputs,
            grad_outputs=torch.ones_like(times),
            create_graph=reuse_grad,
            retain_graph=reuse_grad,
        )[0]

        gradients = torch.einsum(
            "bpij,bpj->bpi", self.inverse_metric(inputs), euc_gradients
        )

        return times.squeeze(-1), gradients

    def gradients(self, inputs, reuse_grad=False):
        """Computes gradient of traveltimes w.r.t. 'xs' and 'xr'.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, 2, dim_signal+1).
            reuse_grad (bool): if true then create_graph=True, retain_graph=True
        Returns:
            gradients (torch.Tensor): The gradients of traveltimes w.r.t. inputs. Shape (batch_size, 2, dim_signal+1).

        """

        _times, gradients = self.times_and_gradients(inputs, reuse_grad=reuse_grad)

        del _times

        return gradients

    def times_grad_vel(
        self,
        inputs,
        reuse_grad=False,
        aux_vel=False,
    ):
        """Computes traveltimes, gradients, and velocities w.r.t. 'xs' and 'xr'.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, 2, dim_signal+1).
            reuse_grad (bool): if true then create_graph=True, retain_graph=True
            aux_vel: If true then also return inverse of velocities.
        Returns:
            times (torch.Tensor): The traveltimes. Shape (batch_size,).
            gradients (torch.Tensor): The gradients of traveltimes w.r.t. inputs. Shape (batch_size, 2, dim_signal+1).
            vel (torch.Tensor): The velocities at input points. Shape (batch_size, 2).
            norm_grad (torch.Tensor): The norm of the gradients. Shape (batch_size, 2).
        """

        times, gradients = self.times_and_gradients(inputs, reuse_grad=reuse_grad)

        epsilon = 1e-12
        norm_grad = torch.sqrt(
            torch.einsum("bpij,bpi,bpj->bp", self.metric(inputs), gradients, gradients)
            + epsilon
        )

        # Calculate velocity with safe reciprocal
        vel = 1.0 / (norm_grad + epsilon)

        if aux_vel:
            return times, gradients, vel, norm_grad
        else:
            return times, gradients, vel

    def velocities(self, inputs):
        """
        Predicted velocity at 'xs' and 'xr.

        Args:
        inputs (torch.Tensor): The pose of the input points. Shape (batch_size, 2, dim_signal+1).
        Returns:
        vel (torch.Tensor): The velocities at input points. Shape (batch_size, 2).
        """

        _times, _grads, vel = self.times_grad_vel(
            inputs, reuse_grad=False, aux_vel=False
        )

        del _times, _grads

        return vel
