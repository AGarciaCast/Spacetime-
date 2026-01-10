from typing import Dict, Union, Optional

import torch
from torch import nn


# Import modules
from toy_experiments.PathFinding.archs import get_backbone


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

        self.xmin = xmin if xmin is not None else [0.0] * (dim_signal + 1)
        self.xmax = xmax if xmax is not None else [1.0] * (dim_signal + 1)

        self.act = nn.GELU()

    def forward(self, inputs):
        """Computes traveltimes.

        Args:
            inputs (torch.Tensor): The input points. Shape (batch_size, 2, dim_signal+1).

        Returns:
        out (torch.Tensor): output.
            Shape (batch_size, num_sample_pairs, 1).
        """

        if self.factored:
            dist = self.distance(inputs)

        if self.normalize_domain:
            # Normalize inputs to [-1, 1]
            inputs = (
                2.0
                * (inputs - torch.tensor(self.xmin, device=inputs.device))
                / (
                    torch.tensor(self.xmax, device=inputs.device)
                    - torch.tensor(self.xmin, device=inputs.device)
                )
                - 1.0
            )

        xs, xr = inputs[:, 0, :], inputs[:, 1, :]

        out_1 = self.backbone(torch.cat([xs, xr], dim=-1))

        out_2 = self.backbone(torch.cat([xr, xs], dim=-1))
        out = 0.5 * (out_1 + out_2)

        out = self.act(out)

        if self.factored:
            out = dist * (1 + out)
        else:
            # pass through relu to ensure positiveness
            out = torch.relu(out)

        return out

    def times(self, inputs, p, a, record_time=False):
        """Computes traveltimes.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p (Tuple[torch.Tensor, torch.Tensor]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, dim_signal). Shape of second component
                (batch_size, num_latents, dim_signal, dim_orientation).
            a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
            gaussian_window_size (float or None): The window size for the gaussian window.

        Returns:
        out (torch.Tensor): output.
            Shape (batch_size, num_sample_pairs).
        """
        outputs = self.forward(inputs, p, a, record_time=record_time)

        if record_time:
            times, forward_time = outputs
            return times.squeeze(-1), forward_time

        else:
            return outputs.squeeze(-1)

    def distance(self, inputs):
        """Homogenous solution, i.e., distance between points.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        """
        raise NotImplementedError("distance method must be implemented.")

    def norm_gradient(self, inputs, gradient):
        """Homogenous solution, i.e., distance between points.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        """
        raise NotImplementedError("distance method must be implemented.")

    def project(self, inputs):
        """Project to manifold

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        """
        raise NotImplementedError("inputs method must be implemented.")

    def gradients(self, inputs, p, a, reuse_grad=False):
        """Computes gradient of traveltimes w.r.t. 'xs' and 'xr'.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p (Tuple[torch.Tensor, torch.Tensor]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, dim_signal). Shape of second component
                (batch_size, num_latents, dim_signal, dim_orientation).
            a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
            gaussian_window_size (float or None): The window size for the gaussian window.
            reuse_grad (bool): if true then create_graph=True, retain_graph=True

        """
        raise NotImplementedError("gradients method must be implemented.")

    def times_and_gradients(self, inputs, p, a, reuse_grad=False, record_time=False):
        """Computes traveltimes, and gradient w.r.t. 'xs' and 'xr' given the time output.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p (Tuple[torch.Tensor, torch.Tensor]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, dim_signal). Shape of second component
                (batch_size, num_latents, dim_signal, dim_orientation).
            a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
            gaussian_window_size (float or None): The window size for the gaussian window.
            reuse_grad (bool): if true then create_graph=True, retain_graph=True

        """
        raise NotImplementedError("times_and_gradients method must be implemented.")

    def times_grad_vel(
        self, inputs, p, a, reuse_grad=False, aux_vel=False, record_time=False
    ):
        """Computes traveltimes, gradients, and velocities w.r.t. 'xs' and 'xr'.

        Args:
            inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p (Tuple[torch.Tensor, torch.Tensor]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, dim_signal). Shape of second component
                (batch_size, num_latents, dim_signal, dim_orientation).
            a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
            gaussian_window_size (float or None): The window size for the gaussian window.
            reuse_grad (bool): if true then create_graph=True, retain_graph=True
            aux_vel (bool): if true then also return inverse of velocities

        """
        raise NotImplementedError("times_grad_vel method must be implemented.")

    def velocities(self, inputs, p, a):
        """
        Predicted velocity at 'xs' and 'xr.

        Args:
        inputs (torch.Tensor): The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        p (Tuple[torch.Tensor, torch.Tensor]): The pose of the latent points. Shape of first component
            (batch_size, num_latents, dim_signal). Shape of second component
            (batch_size, num_latents, dim_signal, dim_orientation).
        a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
        gaussian_window_size (float or None): The window size for the gaussian window.

        """

        _times, _grads, vel = self.times_grad_vel(
            inputs, p, a, reuse_grad=False, aux_vel=False, record_time=False
        )

        del _times, _grads

        return vel
