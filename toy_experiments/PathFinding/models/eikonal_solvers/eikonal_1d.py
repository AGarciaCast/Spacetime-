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
        fourier_emb: Union[None, Dict] = None,
        reparam: Union[None, Dict] = None,
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
            fourier_emb=fourier_emb,
            reparam=reparam,
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

        theta = inputs.requires_grad_(True)
        N = theta.shape[0]

        eta_t, eta_x = self.eta(theta)
        mu_t, mu_x = self.mu(theta)

        eta_vec = torch.stack([eta_t, eta_x], dim=1)
        mu_vec = torch.stack([mu_t, mu_x], dim=1)  #

        I_batch = []
        for n in range(N):
            # Compute Jacobians J_eta and J_mu without breaking autograd.
            grad_eta = []
            grad_mu = []
            for k in range(2):
                grad_eta_k = torch.autograd.grad(
                    eta_vec[n, k], theta, retain_graph=True, create_graph=True
                )[0][
                    n
                ]  # gradient w.r.t. theta for eta_k
                grad_mu_k = torch.autograd.grad(
                    mu_vec[n, k], theta, retain_graph=True, create_graph=True
                )[0][
                    n
                ]  # gradient w.r.t. theta for mu_k
                grad_eta.append(grad_eta_k)
                grad_mu.append(grad_mu_k)

            J_eta = torch.stack(grad_eta, dim=0)
            J_mu = torch.stack(grad_mu, dim=0)
            I_batch.append(J_eta.T @ J_mu)

        out = torch.stack(I_batch, dim=0)
        out = out.view(-1, 2, 2, 2)
        return out

    def mu(self, theta):
        """
        Implementation of the expectation parameter - Eq 22 in the paper. Since our data distribution is 1D, the spacetime is 2D
        Parameters
        ----------
        theta: torch.Tensor
            a batch of spacetime points of shape (N, 2), where the first column is the `time` component, and the second is the `space` component
        Returns
        ----------
        mu_t : torch.Tensor
            `time` component of the expectation parameter - tensor or shape (N,)
        mu_x : torch.Tensor
            `space` component of the expectation parameter - tensor of shape (N,)
        """
        t, x = theta[:, 0], theta[:, 1]
        alpha_t, sigma_t = self.alpha_sigma(t)
        x.requires_grad_(True)
        f = self.eds(t, x)
        div = torch.autograd.grad(f.sum(), x, create_graph=True)[
            0
        ]  # In 1D the divergence is just the derivative
        mu_t, mu_x = sigma_t**2 / alpha_t * div + f**2, f
        return mu_t, mu_x

    def eta(self, theta):
        """
        Implementation of the natural parameter - Eq 18 in the paper.
        Parameters
        ----------
        theta: torch.Tensor
            a batch of spacetime points of shape (N, 2), where the first column is the `time` component, and the second is the `space` component
        Returns
        ----------
        eta_t : torch.Tensor
            `time` component of the natural parameter - tensor or shape (N,)
        eta_x : torch.Tensor
            `space` component of the natural parameter - tensor of shape (N,)
        """
        t, x = theta[:, 0], theta[:, 1]
        alpha_t, sigma_t = self.alpha_sigma(t)
        return -0.5 * alpha_t**2 / sigma_t**2, alpha_t / sigma_t**2 * x

    def eds(self, t, x):
        """Implementation of the denoising mean, or Expected Denoised Sample (EDS) - based on Tweedie formula using the score function - Eq61 in the paper"""
        assert t.shape == x.shape
        assert t.ndim == 1
        alpha_t, sigma_t = self.alpha_sigma(t)
        x.requires_grad_(True)
        log_p_t = self.gaussian_mixture_density(x, t)
        grad_log_p_t = torch.autograd.grad(log_p_t.sum(), x, create_graph=True)[0]
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
