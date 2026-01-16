# Metric Tensor Singularity Test

This script tests whether the metric tensor of the Neural Eikonal Solver 1D is singular for a given geometry configuration.

## Prerequisites

Make sure you have the required dependencies installed:
```bash
# If using conda/mamba
conda activate your_environment

# Or install requirements
pip install torch numpy matplotlib
```

## Usage

### Basic usage
```bash
python test_metric_singularity.py
```

### With options
```bash
# Use GPU
python test_metric_singularity.py --device cuda

# Test with more sample points (creates a 32x32 grid)
python test_metric_singularity.py --num_samples 1024

# Suppress verbose output
python test_metric_singularity.py --no-verbose

# Don't save plots
python test_metric_singularity.py --no-plots

# Custom plot directory
python test_metric_singularity.py --plot_dir ./my_plots
```

## What the script does

1. **Initializes the model** with the specified geometry:
   - `dim_signal`: 1
   - `x_min`: [-1.0, 0.3]
   - `x_max`: [1.0, 0.8]
   - `lambda_min`: -30.0
   - `lambda_max`: 30.0
   - GMM parameters: means=[0.0], variances=[0.5], weights=[1.0]

2. **Samples points** across the spacetime domain (x, t)

3. **Computes the metric tensor** at each point

4. **Analyzes singularities** by checking:
   - Determinants (should be non-zero for non-singular matrices)
   - Eigenvalues (all should be positive for positive-definite metric tensors)
   - Condition numbers (should be reasonable, not extremely large)

5. **Generates visualizations** showing:
   - Determinant distribution
   - Log absolute determinant
   - Minimum eigenvalue
   - Condition number
   - Singularity map

## Output

The script provides:
- **Console output**: Statistics about determinants, eigenvalues, and condition numbers
- **Warning messages**: If singular or ill-conditioned points are detected
- **Plots**: Saved to `metric_singularity_plots/` (or custom directory)

## Understanding the results

### Non-singular metric tensor
- All determinants are non-zero
- All eigenvalues are positive
- Condition numbers are reasonable (< 10^6)

### Singular metric tensor
- Some determinants are (near) zero
- Some eigenvalues are (near) zero
- May indicate numerical issues or physical degeneracies

### Ill-conditioned metric tensor
- Very large condition numbers (> 10^6)
- Numerical operations may be unstable
- Inversion may be inaccurate

## Command-line arguments

- `--device`: Device to run on (`cpu`, `cuda`, or `mps`)
- `--num_samples`: Number of sample points (creates sqrt(n) × sqrt(n) grid)
- `--no-verbose`: Suppress detailed output
- `--no-plots`: Don't save visualization plots
- `--plot_dir`: Directory to save plots (default: `./metric_singularity_plots`)
