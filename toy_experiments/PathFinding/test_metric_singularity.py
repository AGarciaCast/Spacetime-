"""
Script to test if the metric tensor is singular for the Neural Eikonal Solver 1D.

This script:
1. Initializes a NeuralEikonalSolver_1D model with specified geometry
2. Samples points across the spacetime domain
3. Computes the metric tensor at these points
4. Analyzes singularities by computing determinants, eigenvalues, and condition numbers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from toy_experiments.PathFinding.models.eikonal_solvers.eikonal_1d import (
    NeuralEikonalSolver_1D,
)


def test_metric_singularity(
    device="cpu",
    num_samples=100,
    verbose=True,
    save_plots=True,
    plot_dir="./metric_singularity_plots",
):
    """
    Test if the metric tensor is singular for the Neural Eikonal Solver 1D.

    Args:
        device: Device to run on ('cpu', 'cuda', or 'mps')
        num_samples: Number of sample points to test
        verbose: Print detailed information
        save_plots: Save visualization plots
        plot_dir: Directory to save plots
    """

    # Geometry configuration (from your specification)
    # geometry = {
    #     "dim_signal": 1,
    #     "x_min": [-1.0, 0.3],  # space and time min
    #     "x_max": [1.0, 0.7],
    #     "lambda_min": -10.0,
    #     "lambda_max": 10.0,
    #     "gmm_means": [1.0, -1.0],
    #     "gmm_variances": [0.5, 0.5],
    #     "gmm_weights": [0.5, 0.5],
    # }

    geometry = {
        "dim_signal": 1,
        "x_min": [-1.0, 0.3],  # space and time min
        "x_max": [1.0, 0.7],
        "lambda_min": -10.0,
        "lambda_max": 10.0,
        "gmm_means": [0.0],
        "gmm_variances": [0.5],
        "gmm_weights": [1],
    }

    # Solver configuration (reasonable defaults)
    solver_config = {
        "backbone_type": "pirate_net",
        "hidden_dim": 256,
        "num_layers": 4,
        "activation": "ad-gauss-1",
        "nonlinearity": 0.0,
        "use_fourier_features": True,
        "fourier_embed_scale": 1.0,
        "fourier_embed_dim": 256,
        "use_reparam": True,
        "reparam_mean": 1.0,
        "reparam_std": 0.1,
        "factored": True,
        "normalize_domain": True,
    }

    if verbose:
        print("=" * 80)
        print("METRIC TENSOR SINGULARITY TEST")
        print("=" * 80)
        print(f"\nDevice: {device}")
        print(f"Number of test samples: {num_samples}")
        print(f"\nGeometry configuration:")
        for key, value in geometry.items():
            print(f"  {key}: {value}")

    # Initialize the model
    model = NeuralEikonalSolver_1D(
        dim_signal=geometry["dim_signal"],
        backbone_type=solver_config["backbone_type"],
        hidden_dim=solver_config["hidden_dim"],
        num_layers=solver_config["num_layers"],
        original_means=geometry["gmm_means"],
        original_variance=geometry["gmm_variances"],
        weights=geometry["gmm_weights"],
        activation=solver_config["activation"],
        nonlinearity=solver_config["nonlinearity"],
        use_fourier_features=solver_config["use_fourier_features"],
        fourier_embed_scale=solver_config["fourier_embed_scale"],
        fourier_embed_dim=solver_config["fourier_embed_dim"],
        use_reparam=solver_config["use_reparam"],
        reparam_mean=solver_config["reparam_mean"],
        reparam_std=solver_config["reparam_std"],
        xmin=geometry["x_min"],
        xmax=geometry["x_max"],
        factored=solver_config["factored"],
        normalize_domain=solver_config["normalize_domain"],
        lambda_min=geometry["lambda_min"],
        lambda_max=geometry["lambda_max"],
    ).to(device)

    model.eval()

    # Sample points in the domain
    # For 1D signal, the spacetime is 2D: (x, t)
    x_min, t_min = geometry["x_min"]
    x_max, t_max = geometry["x_max"]

    # Create a grid of points
    n_grid = int(np.sqrt(num_samples))
    x_vals = torch.linspace(x_min, x_max, n_grid, device=device)
    t_vals = torch.linspace(t_min, t_max, n_grid, device=device)

    X, T = torch.meshgrid(x_vals, t_vals, indexing="ij")
    points = torch.stack([X.flatten(), T.flatten()], dim=1)  # Shape: (num_samples, 2)

    # Create input pairs (same point for both source and receiver for simplicity)
    # Shape: (num_samples, 2, 2) where middle dim is [source, receiver]
    inputs = points.unsqueeze(1).repeat(1, 2, 1)

    if verbose:
        print(f"\n{'=' * 80}")
        print("COMPUTING METRIC TENSOR")
        print("=" * 80)

    # Compute metric tensor (requires gradients for computation)
    metric_tensor = model.metric_tensor(inputs)

    # Detach to free computation graph and move to CPU for analysis
    metric_tensor = metric_tensor.detach().cpu()
    points = points.cpu()

    # The metric tensor shape is (num_samples, 2, 2, 2)
    # We need to analyze each 2x2 metric tensor
    # Extract the metric tensors (take first index along dimension 1)
    g = metric_tensor[:, 0, :, :]  # Shape: (num_samples, 2, 2)

    if verbose:
        print(f"\nMetric tensor shape: {g.shape}")
        print(f"Expected shape: (num_samples={points.shape[0]}, 2, 2)")

    # Compute determinants
    determinants = torch.det(g)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(g)  # Returns eigenvalues in ascending order
    min_eigenvalues = eigenvalues[:, 0]
    max_eigenvalues = eigenvalues[:, 1]

    # Compute condition numbers
    condition_numbers = max_eigenvalues.abs() / (min_eigenvalues.abs() + 1e-12)

    # Identify singular/nearly singular points
    det_threshold = 1e-6
    eigenvalue_threshold = 1e-6
    condition_threshold = 1e6

    singular_by_det = torch.abs(determinants) < det_threshold
    singular_by_eigenvalue = torch.abs(min_eigenvalues) < eigenvalue_threshold
    ill_conditioned = condition_numbers > condition_threshold

    # Statistics
    num_singular_det = singular_by_det.sum().item()
    num_singular_eig = singular_by_eigenvalue.sum().item()
    num_ill_conditioned = ill_conditioned.sum().item()

    if verbose:
        print(f"\n{'=' * 80}")
        print("SINGULARITY ANALYSIS")
        print("=" * 80)
        print(f"\nDeterminant statistics:")
        print(f"  Min: {determinants.min().item():.6e}")
        print(f"  Max: {determinants.max().item():.6e}")
        print(f"  Mean: {determinants.mean().item():.6e}")
        print(f"  Std: {determinants.std().item():.6e}")
        print(
            f"  Singular points (|det| < {det_threshold}): {num_singular_det}/{points.shape[0]}"
        )

        print(f"\nEigenvalue statistics:")
        print(
            f"  Min eigenvalue range: [{min_eigenvalues.min().item():.6e}, {min_eigenvalues.max().item():.6e}]"
        )
        print(
            f"  Max eigenvalue range: [{max_eigenvalues.min().item():.6e}, {max_eigenvalues.max().item():.6e}]"
        )
        print(
            f"  Singular points (min |eigenvalue| < {eigenvalue_threshold}): {num_singular_eig}/{points.shape[0]}"
        )

        print(f"\nCondition number statistics:")
        print(f"  Min: {condition_numbers.min().item():.6e}")
        print(f"  Max: {condition_numbers.max().item():.6e}")
        print(f"  Mean: {condition_numbers.mean().item():.6e}")
        print(
            f"  Ill-conditioned points (cond > {condition_threshold}): {num_ill_conditioned}/{points.shape[0]}"
        )

        # Print some examples of metric tensors
        print(f"\n{'=' * 80}")
        print("EXAMPLE METRIC TENSORS")
        print("=" * 80)
        for i in range(min(5, points.shape[0])):
            print(f"\nPoint {i}: (x={points[i, 0]:.4f}, t={points[i, 1]:.4f})")
            print(f"Metric tensor:\n{g[i].numpy()}")
            print(f"Determinant: {determinants[i].item():.6e}")
            print(
                f"Eigenvalues: [{eigenvalues[i, 0].item():.6e}, {eigenvalues[i, 1].item():.6e}]"
            )
            print(f"Condition number: {condition_numbers[i].item():.6e}")

    # Create visualizations
    if save_plots:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        # Reshape for plotting
        X_np = X.numpy() if X.device.type == "cpu" else X.cpu().numpy()
        T_np = T.numpy() if T.device.type == "cpu" else T.cpu().numpy()
        det_grid = determinants.reshape(n_grid, n_grid).numpy()
        min_eig_grid = min_eigenvalues.reshape(n_grid, n_grid).numpy()
        cond_grid = condition_numbers.reshape(n_grid, n_grid).numpy()

        # Plot 1: Determinant
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(X_np, T_np, det_grid, levels=50, cmap="RdYlBu")
        plt.colorbar(im, ax=ax, label="Determinant")
        ax.set_xlabel("x (space)")
        ax.set_ylabel("t (time)")
        ax.set_title("Metric Tensor Determinant")
        plt.savefig(f"{plot_dir}/determinant.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2: Log absolute determinant
        fig, ax = plt.subplots(figsize=(10, 8))
        log_det = np.log10(np.abs(det_grid) + 1e-12)
        im = ax.contourf(X_np, T_np, log_det, levels=50, cmap="RdYlBu")
        plt.colorbar(im, ax=ax, label="log10(|Determinant|)")
        ax.set_xlabel("x (space)")
        ax.set_ylabel("t (time)")
        ax.set_title("Log Absolute Determinant of Metric Tensor")
        plt.savefig(f"{plot_dir}/log_determinant.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 3: Minimum eigenvalue
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(X_np, T_np, min_eig_grid, levels=50, cmap="RdYlBu")
        plt.colorbar(im, ax=ax, label="Min Eigenvalue")
        ax.set_xlabel("x (space)")
        ax.set_ylabel("t (time)")
        ax.set_title("Minimum Eigenvalue of Metric Tensor")
        plt.savefig(f"{plot_dir}/min_eigenvalue.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 4: Condition number
        fig, ax = plt.subplots(figsize=(10, 8))
        log_cond = np.log10(cond_grid + 1e-12)
        im = ax.contourf(X_np, T_np, log_cond, levels=50, cmap="RdYlBu_r")
        plt.colorbar(im, ax=ax, label="log10(Condition Number)")
        ax.set_xlabel("x (space)")
        ax.set_ylabel("t (time)")
        ax.set_title("Log Condition Number of Metric Tensor")
        plt.savefig(f"{plot_dir}/condition_number.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 5: Singularity map
        fig, ax = plt.subplots(figsize=(10, 8))
        singular_map = singular_by_det.reshape(n_grid, n_grid).numpy().astype(float)
        im = ax.contourf(X_np, T_np, singular_map, levels=[0, 0.5, 1], cmap="RdYlGn_r")
        plt.colorbar(im, ax=ax, label="Singular (1) / Non-singular (0)")
        ax.set_xlabel("x (space)")
        ax.set_ylabel("t (time)")
        ax.set_title(f"Singularity Map (|det| < {det_threshold})")
        plt.savefig(f"{plot_dir}/singularity_map.png", dpi=150, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Plots saved to: {plot_dir}/")
            print("=" * 80)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)

    is_singular = num_singular_det > 0 or num_singular_eig > 0

    if is_singular:
        print("\n⚠️  WARNING: The metric tensor has SINGULAR points!")
        print(
            f"   - {num_singular_det}/{points.shape[0]} points have |det| < {det_threshold}"
        )
        print(
            f"   - {num_singular_eig}/{points.shape[0]} points have min |eigenvalue| < {eigenvalue_threshold}"
        )
    else:
        print("\n✓ The metric tensor is NON-SINGULAR at all test points.")

    if num_ill_conditioned > 0:
        print(
            f"\n⚠️  WARNING: {num_ill_conditioned}/{points.shape[0]} points are ILL-CONDITIONED (cond > {condition_threshold})"
        )

    print("=" * 80 + "\n")

    return {
        "determinants": determinants,
        "eigenvalues": eigenvalues,
        "condition_numbers": condition_numbers,
        "singular_by_det": singular_by_det,
        "singular_by_eigenvalue": singular_by_eigenvalue,
        "ill_conditioned": ill_conditioned,
        "points": points,
        "metric_tensor": g,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test metric tensor singularity for Neural Eikonal Solver 1D"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of sample points to test (will create sqrt(n) x sqrt(n) grid)",
    )
    parser.add_argument(
        "--no-verbose", action="store_true", help="Suppress detailed output"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Don't save visualization plots"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./metric_singularity_plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    # Auto-select device if cuda/mps available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"

    results = test_metric_singularity(
        device=args.device,
        num_samples=args.num_samples,
        verbose=not args.no_verbose,
        save_plots=not args.no_plots,
        plot_dir=args.plot_dir,
    )
