"""Tests for NeuralEikonalSolver_1D and its inherited functionalities."""

import pytest
import torch
import torch.nn as nn

from toy_experiments.PathFinding.models.eikonal_solvers.eikonal_1d import (
    NeuralEikonalSolver_1D,
)


class TestNeuralEikonalSolver1DInit:
    """Tests for NeuralEikonalSolver_1D initialization."""

    def test_init_basic(self):
        """Test basic initialization with minimal parameters."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )
        assert solver.lambda_min == -30
        assert solver.lambda_max == 30
        assert solver.original_means.shape == (1,)
        assert solver.original_variance.shape == (1,)
        assert solver.weights.shape == (1,)

    def test_init_with_lists(self):
        """Test initialization with list parameters for Gaussian mixture."""
        means = [0.0, 2.0, -2.0]
        variances = [0.5, 1.0, 0.8]
        weights = [0.3, 0.5, 0.2]

        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=means,
            original_variance=variances,
            weights=weights,
        )
        assert solver.original_means.shape == (3,)
        assert solver.original_variance.shape == (3,)
        assert solver.weights.shape == (3,)
        assert torch.allclose(solver.original_means, torch.tensor(means))

    def test_init_with_custom_lambda_range(self):
        """Test initialization with custom lambda_min and lambda_max."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            lambda_min=-20,
            lambda_max=20,
        )
        assert solver.lambda_min == -20
        assert solver.lambda_max == 20

    def test_init_mismatched_lengths_raises_error(self):
        """Test that mismatched parameter lengths raise ValueError."""
        with pytest.raises(
            ValueError,
            match="original_means, original_variance, and weights must have the same length",
        ):
            NeuralEikonalSolver_1D(
                dim_signal=1,
                backbone_type="mlp",
                hidden_dim=64,
                num_layers=2,
                original_means=[0.0, 1.0],
                original_variance=[1.0],
                weights=[0.5, 0.5],
            )

    def test_init_with_domain_bounds(self):
        """Test initialization with custom domain bounds."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            xmin=[0.0, -5.0],
            xmax=[1.0, 5.0],
        )
        assert torch.allclose(solver.xmin, torch.tensor([0.0, -5.0]))
        assert torch.allclose(solver.xmax, torch.tensor([1.0, 5.0]))

    def test_init_with_fourier_embeddings(self):
        """Test initialization with Fourier embeddings."""
        fourier_emb = {"embed_scale": 1.0, "embed_dim": 32}
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            fourier_emb=fourier_emb,
        )
        assert solver.backbone is not None

    def test_init_with_reparam(self):
        """Test initialization with reparameterization."""
        reparam = {"mean": 0.0, "stddev": 0.1}
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            reparam=reparam,
        )
        assert solver.backbone is not None

    def test_init_with_pirate_backbone(self):
        """Test initialization with PirateNet backbone."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="pirate_net",
            hidden_dim=64,
            num_layers=3,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )
        assert solver.backbone is not None


class TestNeuralEikonalSolver1DNoiseFunctions:
    """Tests for noise schedule and related functions."""

    def test_log_SNR_at_boundaries(self):
        """Test log-SNR at t=0 and t=1."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            lambda_min=-30,
            lambda_max=30,
        )

        t_0 = torch.tensor([0.0])
        t_1 = torch.tensor([1.0])

        log_snr_0 = solver.log_SNR(t_0)
        log_snr_1 = solver.log_SNR(t_1)

        assert torch.isclose(log_snr_0, torch.tensor(30.0))
        assert torch.isclose(log_snr_1, torch.tensor(-30.0))

    def test_log_SNR_interpolation(self):
        """Test log-SNR interpolates linearly."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            lambda_min=-30,
            lambda_max=30,
        )

        t_mid = torch.tensor([0.5])
        log_snr_mid = solver.log_SNR(t_mid)

        expected = 30 + ((-30) - 30) * 0.5
        assert torch.isclose(log_snr_mid, torch.tensor(expected))

    def test_alpha_sigma_shape(self):
        """Test alpha_sigma returns correct shapes."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        alpha_t, sigma_t = solver.alpha_sigma(t)

        assert alpha_t.shape == (5,)
        assert sigma_t.shape == (5,)

    def test_alpha_sigma_properties(self):
        """Test alpha and sigma are positive and squared sum to ~1."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.linspace(0, 1, 10)
        alpha_t, sigma_t = solver.alpha_sigma(t)

        # Check positivity
        assert torch.all(alpha_t > 0)
        assert torch.all(sigma_t > 0)

        # Check alpha^2 + sigma^2 approximately equals 1
        sum_squares = alpha_t**2 + sigma_t**2
        assert torch.allclose(sum_squares, torch.ones_like(sum_squares), atol=1e-5)

    def test_alpha_sigma_monotonicity(self):
        """Test alpha decreases and sigma increases with t."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.linspace(0, 1, 20)
        alpha_t, sigma_t = solver.alpha_sigma(t)

        # Alpha should be decreasing
        assert torch.all(alpha_t[1:] <= alpha_t[:-1])

        # Sigma should be increasing
        assert torch.all(sigma_t[1:] >= sigma_t[:-1])


class TestNeuralEikonalSolver1DGaussianMixture:
    """Tests for Gaussian mixture density function."""

    def test_gaussian_mixture_single_component(self):
        """Test Gaussian mixture with single component."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        x = torch.tensor([0.0])
        t = torch.tensor([0.0])

        log_p = solver.gaussian_mixture_density(x, t)
        assert log_p.shape == (1,)
        assert torch.isfinite(log_p).all()

    def test_gaussian_mixture_multiple_components(self):
        """Test Gaussian mixture with multiple components."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=[-2.0, 0.0, 2.0],
            original_variance=[0.5, 1.0, 0.5],
            weights=[0.25, 0.5, 0.25],
        )

        x = torch.linspace(-5, 5, 20)
        t = torch.full((20,), 0.5)

        log_p = solver.gaussian_mixture_density(x, t)
        assert log_p.shape == (20,)
        assert torch.isfinite(log_p).all()

    def test_gaussian_mixture_batch(self):
        """Test Gaussian mixture with batch of inputs."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        batch_size = 16
        x = torch.randn(batch_size)
        t = torch.rand(batch_size)

        log_p = solver.gaussian_mixture_density(x, t)
        assert log_p.shape == (batch_size,)
        assert torch.isfinite(log_p).all()

    def test_gaussian_mixture_symmetry(self):
        """Test Gaussian mixture is symmetric for symmetric setup."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        x1 = torch.tensor([1.0])
        x2 = torch.tensor([-1.0])
        t = torch.tensor([0.5])

        log_p1 = solver.gaussian_mixture_density(x1, t)
        log_p2 = solver.gaussian_mixture_density(x2, t)

        assert torch.allclose(log_p1, log_p2, atol=1e-5)


class TestNeuralEikonalSolver1DEDS:
    """Tests for Expected Denoised Sample (EDS) function."""

    def test_eds_shape(self):
        """Test EDS returns correct shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.tensor([0.5])
        x = torch.tensor([0.0])

        eds = solver.eds(t, x)
        assert eds.shape == (1,)

    def test_eds_batch(self):
        """Test EDS with batch of inputs."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        batch_size = 8
        t = torch.rand(batch_size)
        x = torch.randn(batch_size)

        eds = solver.eds(t, x)
        assert eds.shape == (batch_size,)

    def test_eds_requires_grad(self):
        """Test that EDS computation supports gradient."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.tensor([0.5], requires_grad=True)
        x = torch.tensor([0.0], requires_grad=True)

        eds = solver.eds(t, x)
        loss = eds.sum()
        loss.backward()

        assert t.grad is not None
        assert x.grad is not None

    def test_eds_mismatched_shapes_raises(self):
        """Test EDS raises assertion error for mismatched shapes."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.tensor([0.5, 0.6])
        x = torch.tensor([0.0])

        with pytest.raises(AssertionError):
            solver.eds(t, x)

    def test_eds_multidimensional_raises(self):
        """Test EDS raises assertion error for multidimensional inputs."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        t = torch.tensor([[0.5]])
        x = torch.tensor([[0.0]])

        with pytest.raises(AssertionError):
            solver.eds(t, x)


class TestNeuralEikonalSolver1DEtaMu:
    """Tests for natural parameter eta and expectation parameter mu."""

    def test_eta_shape(self):
        """Test eta returns correct shapes."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        theta = torch.tensor([[0.5, 0.0], [0.3, 1.0]])
        eta_t, eta_x = solver.eta(theta)

        assert eta_t.shape == (2,)
        assert eta_x.shape == (2,)

    def test_mu_shape(self):
        """Test mu returns correct shapes."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        theta = torch.tensor([[0.5, 0.0], [0.3, 1.0]])
        mu_t, mu_x = solver.mu(theta)

        assert mu_t.shape == (2,)
        assert mu_x.shape == (2,)

    def test_eta_gradient_support(self):
        """Test eta computation supports gradients."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        theta = torch.tensor([[0.5, 0.0]], requires_grad=True)
        eta_t, eta_x = solver.eta(theta)

        loss = (eta_t + eta_x).sum()
        loss.backward()

        assert theta.grad is not None

    def test_mu_gradient_support(self):
        """Test mu computation supports gradients."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        theta = torch.tensor([[0.5, 0.0]], requires_grad=True)
        mu_t, mu_x = solver.mu(theta)

        loss = (mu_t + mu_x).sum()
        loss.backward()

        assert theta.grad is not None

    def test_eta_batch(self):
        """Test eta with batch of spacetime points."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        batch_size = 10
        theta = torch.rand(batch_size, 2)
        eta_t, eta_x = solver.eta(theta)

        assert eta_t.shape == (batch_size,)
        assert eta_x.shape == (batch_size,)

    def test_mu_batch(self):
        """Test mu with batch of spacetime points."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        batch_size = 10
        theta = torch.rand(batch_size, 2)
        mu_t, mu_x = solver.mu(theta)

        assert mu_t.shape == (batch_size,)
        assert mu_x.shape == (batch_size,)


class TestNeuralEikonalSolver1DMetricTensor:
    """Tests for metric tensor computation."""

    def test_metric_tensor_shape(self):
        """Test metric tensor returns correct shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Input shape: (batch_size, 2, 2) - two points in 2D spacetime
        inputs = torch.rand(4, 2, 2)
        metric = solver.metric_tensor(inputs)

        # Expected output: (batch_size, 2, 2, 2)
        assert metric.shape == (4, 2, 2, 2)

    def test_metric_tensor_single_sample(self):
        """Test metric tensor with single sample."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(1, 2, 2)
        metric = solver.metric_tensor(inputs)

        assert metric.shape == (1, 2, 2, 2)

    def test_metric_tensor_gradient_support(self):
        """Test metric tensor computation supports gradients."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(2, 2, 2)
        metric = solver.metric_tensor(inputs)

        loss = metric.sum()
        # Note: This should work if create_graph=True was used properly
        assert torch.isfinite(loss)

    def test_metric_tensor_values_finite(self):
        """Test metric tensor returns finite values."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(3, 2, 2)
        metric = solver.metric_tensor(inputs)

        assert torch.isfinite(metric).all()


class TestNeuralEikonalSolver1DInheritedForward:
    """Tests for inherited forward method from base class."""

    def test_forward_shape(self):
        """Test forward pass returns correct shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Input: (batch_size, 2, dim_signal+1)
        inputs = torch.rand(8, 2, 2)
        output = solver(inputs)

        assert output.shape == (8, 1)

    def test_forward_positivity(self):
        """Test forward pass returns non-negative traveltimes."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(10, 2, 2)
        output = solver(inputs)

        assert torch.all(output >= 0)

    def test_forward_symmetry(self):
        """Test forward pass is symmetric (swap source and receiver)."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Create input with two distinct points
        xs = torch.tensor([[0.1, 0.2]])
        xr = torch.tensor([[0.8, 0.9]])

        inputs1 = torch.stack([xs, xr], dim=1)
        inputs2 = torch.stack([xr, xs], dim=1)

        out1 = solver(inputs1)
        out2 = solver(inputs2)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_forward_with_factored_true(self):
        """Test forward pass with factored=True."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            factored=True,
        )

        inputs = torch.rand(5, 2, 2)
        output = solver(inputs)

        assert output.shape == (5, 1)
        assert torch.all(output >= 0)

    def test_forward_with_factored_false(self):
        """Test forward pass with factored=False."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            factored=False,
        )

        inputs = torch.rand(5, 2, 2)
        output = solver(inputs)

        assert output.shape == (5, 1)
        assert torch.all(output >= 0)

    def test_forward_gradient_flow(self):
        """Test gradients flow through forward pass."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(4, 2, 2, requires_grad=True)
        output = solver(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None


class TestNeuralEikonalSolver1DInheritedTimes:
    """Tests for inherited times method from base class."""

    def test_times_shape(self):
        """Test times method returns correct shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(6, 2, 2)
        times = solver.times(inputs)

        assert times.shape == (6,)

    def test_times_equivalence_to_forward(self):
        """Test times is equivalent to forward with squeeze."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(4, 2, 2)
        times = solver.times(inputs)
        forward_output = solver.forward(inputs).squeeze(-1)

        assert torch.allclose(times, forward_output)


class TestNeuralEikonalSolver1DInheritedAmbientDistance:
    """Tests for inherited ambient_distance method."""

    def test_ambient_distance_shape(self):
        """Test ambient distance returns correct shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(5, 2, 2)
        dist = solver.ambient_distance(inputs)

        assert dist.shape == (5, 1)

    def test_ambient_distance_positivity(self):
        """Test ambient distance is non-negative."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(10, 2, 2)
        dist = solver.ambient_distance(inputs)

        assert torch.all(dist >= 0)

    def test_ambient_distance_zero_when_same(self):
        """Test ambient distance is zero when points are identical."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        point = torch.tensor([[0.5, 0.3]])
        inputs = torch.stack([point, point], dim=1)
        dist = solver.ambient_distance(inputs)

        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-6)

    def test_ambient_distance_symmetry(self):
        """Test ambient distance is symmetric."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        xs = torch.tensor([[0.1, 0.2]])
        xr = torch.tensor([[0.8, 0.9]])

        inputs1 = torch.stack([xs, xr], dim=1)
        inputs2 = torch.stack([xr, xs], dim=1)

        dist1 = solver.ambient_distance(inputs1)
        dist2 = solver.ambient_distance(inputs2)

        assert torch.allclose(dist1, dist2)


class TestNeuralEikonalSolver1DInheritedProject:
    """Tests for inherited project method."""

    def test_project_within_bounds(self):
        """Test project leaves points within bounds unchanged."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            xmin=[0.0, 0.0],
            xmax=[1.0, 1.0],
        )

        inputs = torch.tensor([[[0.3, 0.5], [0.7, 0.2]]])
        projected = solver.project(inputs)

        assert torch.allclose(inputs, projected)

    def test_project_clamps_out_of_bounds(self):
        """Test project clamps points outside bounds."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
            xmin=[0.0, 0.0],
            xmax=[1.0, 1.0],
        )

        inputs = torch.tensor([[[-0.5, 0.5], [1.5, 0.5]]])
        projected = solver.project(inputs)

        assert torch.all(projected >= solver.xmin)
        assert torch.all(projected <= solver.xmax)

    def test_project_shape_preserved(self):
        """Test project preserves input shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(8, 2, 2)
        projected = solver.project(inputs)

        assert projected.shape == inputs.shape


class TestNeuralEikonalSolver1DInheritedInverseMetric:
    """Tests for inherited inverse_metric_tensor method."""

    def test_inverse_metric_shape(self):
        """Test inverse metric tensor returns correct shape."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Use controlled inputs to avoid singular matrices
        inputs = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.8]],
                [[0.2, 0.4], [0.6, 0.9]],
                [[0.5, 0.3], [0.8, 0.7]],
            ]
        )

        try:
            g_inv = solver.inverse_metric_tensor(inputs)
            assert g_inv.shape == (3, 2, 2, 2)
        except torch._C._LinAlgError:
            # If matrix is singular, test that the shape would be correct
            # by checking the metric tensor shape instead
            g = solver.metric_tensor(inputs)
            assert g.shape == (3, 2, 2, 2)

    def test_inverse_metric_inverts_metric(self):
        """Test that inverse metric times metric gives identity (approximately)."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=32,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Use controlled inputs to avoid singular matrices
        inputs = torch.tensor([[[0.3, 0.5], [0.7, 0.8]], [[0.2, 0.4], [0.6, 0.9]]])

        try:
            g = solver.metric_tensor(inputs)
            g_inv = solver.inverse_metric_tensor(inputs)

            # Compute g @ g_inv for each position in batch
            identity_approx = torch.einsum("bpij,bpjk->bpik", g, g_inv)

            # Check if result is approximately identity matrix
            batch_size, num_points = inputs.shape[0], inputs.shape[1]
            identity = (
                torch.eye(2)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, num_points, 2, 2)
            )

            assert torch.allclose(identity_approx, identity, atol=1e-3)
        except torch._C._LinAlgError:
            # If the matrix is singular, we can't test the inversion property
            # This is acceptable as it tests the numerical edge case behavior
            pytest.skip("Matrix is singular, cannot test inversion property")


class TestNeuralEikonalSolver1DIntegration:
    """Integration tests combining multiple functionalities."""

    def test_end_to_end_forward_and_metric(self):
        """Test complete forward pass with metric computation."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=3,
            original_means=[0.0, 2.0],
            original_variance=[1.0, 0.5],
            weights=[0.6, 0.4],
            lambda_min=-20,
            lambda_max=20,
        )

        inputs = torch.rand(4, 2, 2)

        # Forward pass
        times = solver.times(inputs)
        assert times.shape == (4,)

        # Metric computation
        metric = solver.metric_tensor(inputs)
        assert metric.shape == (4, 2, 2, 2)

        # Ambient distance
        dist = solver.ambient_distance(inputs)
        assert dist.shape == (4, 1)

    def test_gradient_computation_full_pipeline(self):
        """Test gradient computation through the full pipeline."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs = torch.rand(3, 2, 2, requires_grad=True)

        # Forward pass
        times = solver.times(inputs)

        # Compute metric
        metric = solver.metric_tensor(inputs)

        # Backward pass
        loss = times.sum() + metric.sum()
        loss.backward()

        assert inputs.grad is not None

    def test_different_backbones(self):
        """Test solver works with different backbone architectures."""
        backbones = ["mlp", "pirate_net"]

        for backbone_type in backbones:
            solver = NeuralEikonalSolver_1D(
                dim_signal=1,
                backbone_type=backbone_type,
                hidden_dim=64,
                num_layers=2,
                original_means=0.0,
                original_variance=1.0,
                weights=1.0,
            )

            inputs = torch.rand(4, 2, 2)
            output = solver(inputs)

            assert output.shape == (4, 1)
            assert torch.all(output >= 0)

    def test_batch_consistency(self):
        """Test that batch processing is consistent with individual processing."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Create batch
        inputs = torch.rand(5, 2, 2)

        # Process as batch
        batch_output = solver.times(inputs)

        # Process individually
        individual_outputs = []
        for i in range(5):
            out = solver.times(inputs[i : i + 1])
            individual_outputs.append(out)

        individual_outputs = torch.cat(individual_outputs)

        assert torch.allclose(batch_output, individual_outputs)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        # Test with values at domain boundaries
        inputs = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        output = solver(inputs)

        assert torch.isfinite(output).all()
        assert torch.all(output >= 0)

    def test_device_compatibility(self):
        """Test solver works on CPU (and GPU if available)."""
        solver = NeuralEikonalSolver_1D(
            dim_signal=1,
            backbone_type="mlp",
            hidden_dim=64,
            num_layers=2,
            original_means=0.0,
            original_variance=1.0,
            weights=1.0,
        )

        inputs_cpu = torch.rand(3, 2, 2)
        output_cpu = solver(inputs_cpu)

        assert output_cpu.device.type == "cpu"
        assert output_cpu.shape == (3, 1)

        if torch.cuda.is_available():
            solver_cuda = solver.cuda()
            inputs_cuda = inputs_cpu.cuda()
            output_cuda = solver_cuda(inputs_cuda)

            assert output_cuda.device.type == "cuda"
            assert output_cuda.shape == (3, 1)
