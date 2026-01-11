"""Tests for neural architectures in archs.py."""

import pytest
import torch
import torch.nn as nn

from toy_experiments.PathFinding.models.archs import (
    FourierEmbs,
    ReparamDense,
    MLP,
    PIModifiedBottleneck,
    PirateNet,
)


class TestFourierEmbs:
    """Tests for FourierEmbs module."""

    def test_init_valid_embed_dim(self):
        """Test FourierEmbs initialization with valid even embed_dim."""
        emb = FourierEmbs(embed_scale=1.0, embed_dim=64)
        assert emb.embed_scale == 1.0
        assert emb.embed_dim == 64
        assert emb.kernel is None

    def test_init_invalid_embed_dim(self):
        """Test FourierEmbs raises error with odd embed_dim."""
        with pytest.raises(ValueError, match="embed_dim must be even"):
            FourierEmbs(embed_scale=1.0, embed_dim=63)

    def test_forward_shape(self):
        """Test FourierEmbs forward pass output shape."""
        emb = FourierEmbs(embed_scale=1.0, embed_dim=64)
        x = torch.randn(8, 2)
        y = emb(x)
        assert y.shape == (8, 64)

    def test_forward_kernel_initialization(self):
        """Test that kernel is initialized on first forward pass."""
        emb = FourierEmbs(embed_scale=1.0, embed_dim=64)
        assert emb.kernel is None
        x = torch.randn(8, 2)
        emb(x)
        assert emb.kernel is not None
        assert emb.kernel.shape == (2, 32)

    def test_forward_kernel_reuse(self):
        """Test that kernel is reused across forward passes."""
        emb = FourierEmbs(embed_scale=1.0, embed_dim=64)
        x = torch.randn(8, 2)
        emb(x)
        kernel1 = emb.kernel
        emb(x)
        kernel2 = emb.kernel
        assert kernel1 is kernel2

    def test_forward_different_batch_sizes(self):
        """Test FourierEmbs with different batch sizes."""
        emb = FourierEmbs(embed_scale=2.0, embed_dim=32)
        x1 = torch.randn(4, 3)
        x2 = torch.randn(16, 3)
        y1 = emb(x1)
        y2 = emb(x2)
        assert y1.shape == (4, 32)
        assert y2.shape == (16, 32)


class TestReparamDense:
    """Tests for ReparamDense module."""

    def test_init_without_reparam(self):
        """Test ReparamDense initialization without reparameterization."""
        layer = ReparamDense(in_features=10, out_features=5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.linear is not None
        assert layer.g is None
        assert layer.v is None

    def test_init_with_reparam(self):
        """Test ReparamDense initialization with weight factorization."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        layer = ReparamDense(in_features=10, out_features=5, reparam=reparam)
        assert layer.linear is None
        assert layer.g is not None
        assert layer.v is not None
        assert layer.g.shape == (5,)
        assert layer.v.shape == (10, 5)

    def test_init_invalid_reparam_type(self):
        """Test ReparamDense raises error with unsupported reparam type."""
        reparam = {"type": "invalid_type"}
        with pytest.raises(ValueError, match="Unsupported reparam type"):
            ReparamDense(in_features=10, out_features=5, reparam=reparam)

    def test_forward_without_reparam(self):
        """Test ReparamDense forward pass without reparameterization."""
        layer = ReparamDense(in_features=10, out_features=5)
        x = torch.randn(8, 10)
        y = layer(x)
        assert y.shape == (8, 5)

    def test_forward_with_reparam(self):
        """Test ReparamDense forward pass with reparameterization."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        layer = ReparamDense(in_features=10, out_features=5, reparam=reparam)
        x = torch.randn(8, 10)
        y = layer(x)
        assert y.shape == (8, 5)

    def test_forward_with_bias(self):
        """Test ReparamDense forward pass with bias."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        layer = ReparamDense(in_features=10, out_features=5, reparam=reparam, bias=True)
        assert layer.bias is not None
        x = torch.randn(8, 10)
        y = layer(x)
        assert y.shape == (8, 5)

    def test_forward_without_bias(self):
        """Test ReparamDense forward pass without bias."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        layer = ReparamDense(
            in_features=10, out_features=5, reparam=reparam, bias=False
        )
        assert layer.bias is None
        x = torch.randn(8, 10)
        y = layer(x)
        assert y.shape == (8, 5)


class TestMLP:
    """Tests for MLP module."""

    def test_init_basic(self):
        """Test MLP initialization with basic parameters."""
        mlp = MLP(input_dim=10, hidden_dim=64, num_layers=3, out_dim=1)
        assert len(mlp.layers) == 4  # 3 hidden + 1 output
        assert len(mlp.activations) == 3

    def test_init_with_list_hidden_dim(self):
        """Test MLP initialization with list of hidden dimensions."""
        mlp = MLP(input_dim=10, hidden_dim=[64, 128, 64], num_layers=3, out_dim=2)
        assert len(mlp.layers) == 4
        assert mlp.layers[0].out_features == 64
        assert mlp.layers[1].out_features == 128
        assert mlp.layers[2].out_features == 64
        assert mlp.layers[3].out_features == 2

    def test_init_with_fourier_emb(self):
        """Test MLP initialization with Fourier embeddings."""
        fourier_emb = {"embed_scale": 1.0, "embed_dim": 32}
        mlp = MLP(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            out_dim=1,
            fourier_emb=fourier_emb,
        )
        assert mlp.embedding is not None
        assert isinstance(mlp.embedding, FourierEmbs)

    def test_init_with_reparam(self):
        """Test MLP initialization with reparameterization."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        mlp = MLP(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            out_dim=1,
            reparam=reparam,
        )
        assert mlp.layers[0].reparam is not None

    def test_forward_shape(self):
        """Test MLP forward pass output shape."""
        mlp = MLP(input_dim=10, hidden_dim=64, num_layers=3, out_dim=2)
        x = torch.randn(8, 10)
        y = mlp(x)
        assert y.shape == (8, 2)

    def test_forward_with_fourier_emb(self):
        """Test MLP forward pass with Fourier embeddings."""
        fourier_emb = {"embed_scale": 1.0, "embed_dim": 32}
        mlp = MLP(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            out_dim=1,
            fourier_emb=fourier_emb,
        )
        x = torch.randn(8, 10)
        y = mlp(x)
        assert y.shape == (8, 1)

    def test_forward_different_activations(self):
        """Test MLP with different activation functions."""
        activations = ["tanh", "relu", "ad-gauss-1", "sin"]
        for act in activations:
            mlp = MLP(
                input_dim=10,
                hidden_dim=64,
                num_layers=2,
                out_dim=1,
                activation=act,
            )
            x = torch.randn(4, 10)
            y = mlp(x)
            assert y.shape == (4, 1)

    def test_forward_different_out_activations(self):
        """Test MLP with different output activations."""
        out_activations = ["linear", "tanh", "sigmoid"]
        for out_act in out_activations:
            mlp = MLP(
                input_dim=10,
                hidden_dim=64,
                num_layers=2,
                out_dim=1,
                out_activation=out_act,
            )
            x = torch.randn(4, 10)
            y = mlp(x)
            assert y.shape == (4, 1)


class TestPIModifiedBottleneck:
    """Tests for PIModifiedBottleneck module."""

    def test_init(self):
        """Test PIModifiedBottleneck initialization."""
        block = PIModifiedBottleneck(
            input_dim=32,
            hidden_dim=64,
            activation="tanh",
            nonlinearity=0.5,
            reparam=None,
        )
        assert block.input_dim == 32
        assert block.hidden_dim == 64
        assert block.alpha is not None

    def test_forward_shape(self):
        """Test PIModifiedBottleneck forward pass output shape."""
        block = PIModifiedBottleneck(
            input_dim=32,
            hidden_dim=64,
            activation="tanh",
            nonlinearity=0.5,
            reparam=None,
        )
        x = torch.randn(8, 32)
        u = torch.randn(8, 64)
        v = torch.randn(8, 64)
        y = block(x, u, v)
        assert y.shape == (8, 32)

    def test_forward_with_reparam(self):
        """Test PIModifiedBottleneck forward pass with reparameterization."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        block = PIModifiedBottleneck(
            input_dim=32,
            hidden_dim=64,
            activation="tanh",
            nonlinearity=0.5,
            reparam=reparam,
        )
        x = torch.randn(8, 32)
        u = torch.randn(8, 64)
        v = torch.randn(8, 64)
        y = block(x, u, v)
        assert y.shape == (8, 32)

    def test_skip_connection(self):
        """Test that skip connection preserves input dimension."""
        block = PIModifiedBottleneck(
            input_dim=32,
            hidden_dim=64,
            activation="tanh",
            nonlinearity=0.0,  # No mixing with transformed output
            reparam=None,
        )
        x = torch.randn(8, 32)
        u = torch.randn(8, 64)
        v = torch.randn(8, 64)
        y = block(x, u, v)
        # With alpha=0, output should be close to identity
        assert y.shape == x.shape


class TestPirateNet:
    """Tests for PirateNet module."""

    def test_init_basic(self):
        """Test PirateNet initialization with basic parameters."""
        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=1,
        )
        assert net.input_dim == 10
        assert net.num_layers == 2
        assert net.hidden_dim == 64
        assert net.out_dim == 1
        assert len(net.blocks) == 2

    def test_init_with_fourier_emb(self):
        """Test PirateNet initialization with Fourier embeddings."""
        fourier_emb = {"embed_scale": 1.0, "embed_dim": 32}
        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=1,
            fourier_emb=fourier_emb,
        )
        assert net.embedding is not None
        assert isinstance(net.embedding, FourierEmbs)

    def test_init_with_reparam(self):
        """Test PirateNet initialization with reparameterization."""
        reparam = {"type": "weight_fact", "mean": 0.0, "stddev": 0.1}
        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=1,
            reparam=reparam,
        )
        assert net.u_proj.reparam is not None
        assert net.v_proj.reparam is not None

    def test_forward_shape(self):
        """Test PirateNet forward pass output shape."""
        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=1,
        )
        x = torch.randn(8, 10)
        y = net(x)
        assert y.shape == (8, 1)

    def test_forward_with_fourier_emb(self):
        """Test PirateNet forward pass with Fourier embeddings."""
        fourier_emb = {"embed_scale": 1.0, "embed_dim": 32}
        net = PirateNet(
            input_dim=10,
            num_layers=3,
            hidden_dim=64,
            out_dim=2,
            fourier_emb=fourier_emb,
        )
        x = torch.randn(8, 10)
        y = net(x)
        assert y.shape == (8, 2)

    def test_forward_different_layers(self):
        """Test PirateNet with different numbers of layers."""
        for num_layers in [1, 2, 4, 8]:
            net = PirateNet(
                input_dim=10,
                num_layers=num_layers,
                hidden_dim=64,
                out_dim=1,
            )
            x = torch.randn(4, 10)
            y = net(x)
            assert y.shape == (4, 1)
            assert len(net.blocks) == num_layers

    def test_forward_different_activations(self):
        """Test PirateNet with different activation functions."""
        activations = ["tanh", "relu", "ad-gauss-1", "sin"]
        for act in activations:
            net = PirateNet(
                input_dim=10,
                num_layers=2,
                hidden_dim=64,
                out_dim=1,
                activation=act,
            )
            x = torch.randn(4, 10)
            y = net(x)
            assert y.shape == (4, 1)

    def test_embedding_dim_without_fourier(self):
        """Test _embedding_dim method without Fourier embeddings."""
        net = PirateNet(input_dim=10, num_layers=2, hidden_dim=64, out_dim=1)
        assert net._embedding_dim() == 10

    def test_embedding_dim_with_fourier(self):
        """Test _embedding_dim method with Fourier embeddings."""
        fourier_emb = {"embed_scale": 1.0, "embed_dim": 32}
        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=1,
            fourier_emb=fourier_emb,
        )
        assert net._embedding_dim() == 32

    def test_forward_multi_output(self):
        """Test PirateNet with multiple output dimensions."""
        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=5,
        )
        x = torch.randn(8, 10)
        y = net(x)
        assert y.shape == (8, 5)


class TestIntegration:
    """Integration tests for model combinations."""

    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP."""
        mlp = MLP(input_dim=10, hidden_dim=64, num_layers=2, out_dim=1)
        x = torch.randn(4, 10, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert mlp.layers[0].linear.weight.grad is not None

    def test_piratenet_gradient_flow(self):
        """Test that gradients flow through PirateNet."""
        net = PirateNet(input_dim=10, num_layers=2, hidden_dim=64, out_dim=1)
        x = torch.randn(4, 10, requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert net.u_proj.linear.weight.grad is not None

    def test_models_with_same_config(self):
        """Test MLP and PirateNet with similar configurations."""
        x = torch.randn(8, 10)

        mlp = MLP(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            out_dim=1,
            activation="tanh",
        )

        net = PirateNet(
            input_dim=10,
            num_layers=2,
            hidden_dim=64,
            out_dim=1,
            activation="tanh",
        )

        y_mlp = mlp(x)
        y_net = net(x)

        assert y_mlp.shape == y_net.shape == (8, 1)
