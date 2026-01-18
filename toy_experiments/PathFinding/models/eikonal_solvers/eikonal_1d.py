from typing import Dict, Union
import torch
from torch import nn

# Import modules
from toy_experiments.PathFinding.models.eikonal_solvers._base_eikonal_solver import (
    NeuralEikonalSolver,
)


class NeuralEikonalSolver_1D(NeuralEikonalSolver):
    """1D  Neural Eikonal Solver."""

    def __init__(
        self,
        dim_signal: int,
        backbone_type: str,
        hidden_dim: int,
        num_layers: int,
        original_means,
        original_variance,
        weights,
        activation="ad-gauss-1",
        nonlinearity: float = 0.0,
        use_fourier_features: bool = False,
        fourier_embed_scale: float = 10.0,
        fourier_embed_dim: int = 256,
        use_reparam: bool = False,
        reparam_mean: float = 0.0,
        reparam_std: float = 0.1,
        xmin=None,
        xmax=None,
        factored: bool = True,
        normalize_domain: bool = True,
        lambda_min: float = -30,
        lambda_max: float = 30,
    ):
        super().__init__(
            dim_signal=dim_signal,
            backbone_type=backbone_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            nonlinearity=nonlinearity,
            use_fourier_features=use_fourier_features,
            fourier_embed_scale=fourier_embed_scale,
            fourier_embed_dim=fourier_embed_dim,
            use_reparam=use_reparam,
            reparam_mean=reparam_mean,
            reparam_std=reparam_std,
            xmin=xmin,
            xmax=xmax,
            factored=factored,
            normalize_domain=normalize_domain,
        )

        assert dim_signal == 1, "NeuralEikonalSolver_1D only supports 1D signals."

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        if type(original_means) is int or type(original_means) is float:
            original_means = [original_means]
        if type(original_variance) is int or type(original_variance) is float:
            original_variance = [original_variance]
        if type(weights) is int or type(weights) is float:
            weights = [weights]

        if not (len(original_means) == len(original_variance) == len(weights)):
            raise ValueError(
                "original_means, original_variance, and weights must have the same length."
            )

        self.register_buffer(
            "original_means", torch.tensor(original_means, dtype=torch.float32)
        )
        self.register_buffer(
            "original_variance", torch.tensor(original_variance, dtype=torch.float32)
        )
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def metric_tensor(self, inputs):
        # inputs shape: (N, 2, 2)
        # map to (Nx2, 2)
        inputs = inputs.view(-1, 2)

        # Detach to keep metric computation independent from the training graph.
        theta = inputs.detach().requires_grad_(True)

        eta_t, eta_x = self.eta(theta)
        mu_t, mu_x = self.mu(theta)

        eta_vec = torch.stack([eta_t, eta_x], dim=1)  # (N, 2)
        mu_vec = torch.stack([mu_t, mu_x], dim=1)  # (N, 2)

        # Vectorized Jacobian computation
        # For eta_vec: compute gradients for both components simultaneously
        J_eta_list = []
        for k in range(2):
            grad_k = torch.autograd.grad(
                eta_vec[:, k].sum(),
                theta,
                retain_graph=True,
                create_graph=False,
            )[
                0
            ]  # (N, 2)
            J_eta_list.append(grad_k)
        J_eta = torch.stack(
            J_eta_list, dim=1
        )  # (N, 2, 2) - [batch, output_dim, input_dim]

        # For mu_vec: compute gradients for both components simultaneously
        J_mu_list = []
        for k in range(2):
            grad_k = torch.autograd.grad(
                mu_vec[:, k].sum(),
                theta,
                retain_graph=True,
                create_graph=False,
            )[
                0
            ]  # (N, 2)
            J_mu_list.append(grad_k)
        J_mu = torch.stack(
            J_mu_list, dim=1
        )  # (N, 2, 2) - [batch, output_dim, input_dim]

        # Compute J_eta.T @ J_mu for all batch elements at once
        # J_eta.T is (N, 2, 2) with dimensions [batch, input_dim, output_dim]
        # J_mu is (N, 2, 2) with dimensions [batch, output_dim, input_dim]
        # Result should be (N, 2, 2)
        out = torch.einsum("nij,njk->nik", J_eta.transpose(-2, -1), J_mu)  # (N, 2, 2)

        # Reshape to match expected output
        out = out.view(-1, 2, 2, 2)

        return out

    def mu(self, theta):
        """
        Implementation of the expectation parameter - Eq 22 in the paper. Since our data distribution is 1D, the spacetime is 2D
        Parameters
        ----------
        theta: torch.Tensor
            a batch of spacetime points of shape (N, 2), where the first column is the `space` component, and the second is the `time` component
        Returns
        ----------
        mu_t : torch.Tensor
            `time` component of the expectation parameter - tensor or shape (N,)
        mu_x : torch.Tensor
            `space` component of the expectation parameter - tensor of shape (N,)
        """
        x, t = theta[:, 0], theta[:, 1]
        alpha_t, sigma_t = self.alpha_sigma(t)
        f = self.eds(t, x)
        div = torch.autograd.grad(
            f.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]
        mu_t, mu_x = sigma_t**2 / alpha_t * div + f**2, f
        return mu_t, mu_x

    def eta(self, theta):
        """
        Implementation of the natural parameter - Eq 18 in the paper.
        Parameters
        ----------
        theta: torch.Tensor
            a batch of spacetime points of shape (N, 2), where the first column is the `space` component, and the second is the `time` component
        Returns
        ----------
        eta_t : torch.Tensor
            `time` component of the natural parameter - tensor or shape (N,)
        eta_x : torch.Tensor
            `space` component of the natural parameter - tensor of shape (N,)
        """
        x, t = theta[:, 0], theta[:, 1]
        alpha_t, sigma_t = self.alpha_sigma(t)
        return -0.5 * alpha_t**2 / sigma_t**2, alpha_t / sigma_t**2 * x

    def eds(self, t, x):
        """Implementation of the denoising mean, or Expected Denoised Sample (EDS) - based on Tweedie formula using the score function - Eq61 in the paper"""
        assert t.shape == x.shape
        assert t.ndim == 1
        alpha_t, sigma_t = self.alpha_sigma(t)
        log_p_t = self.gaussian_mixture_density(x, t)
        grad_log_p_t = torch.autograd.grad(
            log_p_t.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]
        res = x + sigma_t**2 * grad_log_p_t
        res = res / alpha_t
        return res

    def log_SNR(self, t):
        """Implementation of the linear-logSNR noise schedule"""
        return self.lambda_max + (self.lambda_min - self.lambda_max) * t

    def alpha_sigma(self, t):
        lambda_t = self.log_SNR(t)
        alpha_t = torch.sigmoid(lambda_t).sqrt()
        sigma_t = torch.sigmoid(-lambda_t).sqrt()
        return alpha_t, sigma_t

    def gaussian_mixture_density(self, x, t):
        """Analytical implementation of the marginal log-density at time t"""
        alpha_t, sigma_t = self.alpha_sigma(t)
        means_t = alpha_t[:, None] * self.original_means[None, :]

        variances_t = (
            sigma_t[:, None] ** 2 + alpha_t[:, None] ** 2 * self.original_variance
        )
        log_probs = torch.log(self.weights[None, :]) - 0.5 * (
            torch.log(2 * torch.pi * variances_t)
            + (x[:, None] - means_t) ** 2 / variances_t
        )
        log_p_t = torch.logsumexp(log_probs, dim=1)
        return log_p_t

    def sample_trajectory_points(
        self,
        n_trajectories=100,
        n_steps=512,
        n_points_per_trajectory=10,
        x_range=(-2, 2),
        t_start=1.0,
        t_end=0,
    ):
        """
        Sample points along multiple PF-ODE trajectories.

        Parameters:
        -----------
        n_trajectories : int
            Number of different trajectories to generate
        n_steps : int
            Number of integration steps per trajectory
        n_points_per_trajectory : int
            Number of points to sample from each trajectory
        x_range : tuple
            Range of initial x values to sample from
        t_start, t_end : float
            Start and end times for the ODE integration
        seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        sampled_points : torch.Tensor
            Shape (n_trajectories * n_points_per_trajectory, 2) where each row is (t, x)
        trajectory_ids : torch.Tensor
            Shape (n_trajectories * n_points_per_trajectory,) indicating which trajectory each point came from
        """

        all_sampled_points = []

        for traj_id in range(n_trajectories):
            x_init = torch.empty(
                1, dtype=torch.float32, device=self.original_means.device
            ).uniform_(*x_range)
            t_trajectory, x_trajectory = self.sample(
                x_init, n_steps, t_start, t_end
            )

            # Combine into spacetime points (x, t)
            trajectory_points = torch.stack(
                [x_trajectory.flatten(), t_trajectory.flatten()], dim=1
            )

            # Sample n_points_per_trajectory random points from this trajectory
            if len(trajectory_points) >= n_points_per_trajectory:
                indices = torch.randperm(
                    len(trajectory_points), device=trajectory_points.device
                )[:n_points_per_trajectory]
                sampled_points = trajectory_points[indices]
            else:
                sampled_points = trajectory_points

            all_sampled_points.append(sampled_points)

        sampled_points = torch.cat(all_sampled_points, dim=0)

        return sampled_points.detach().cpu()

    def sample(self, x, n_steps, t_start=1, t_end=0):
        """PF-ODE sampling"""
        t = t_start * torch.ones_like(x)
        dt_val = (t_start - t_end) / n_steps
        all_x = [x.detach().clone()]
        all_t = [t.detach().clone()]
        for i in range(n_steps):
            dt, dx = self.compute_vector_field(x, t)
            x = x + dt * dx * dt_val
            t = t + dt * dt_val
            all_x.append(x.detach().clone())
            all_t.append(t.detach().clone())
        return torch.stack(all_t, dim=0), torch.stack(all_x, dim=0)

    def compute_vector_field(self, x, t):
        """Implementation of the PF-ODE vector field"""
        alpha_t, sigma_t = self.alpha_sigma(t)
        f_t = 0.5 * (self.lambda_min - self.lambda_max) * sigma_t**2
        g2_t = (self.lambda_max - self.lambda_min) * sigma_t**2

        x.requires_grad_(True)
        log_p_t = self.gaussian_mixture_density(x, t)
        grad_log_p_t = torch.autograd.grad(log_p_t.sum(), x, create_graph=True)[0]

        dx = f_t * x - 0.5 * g2_t * grad_log_p_t
        dt = -torch.ones_like(dx)
        return dt.detach(), dx.detach()
