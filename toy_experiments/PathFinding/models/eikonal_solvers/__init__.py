from toy_experiments.PathFinding.models.eikonal_solvers.eikonal_1d import (
    NeuralEikonalSolver_1D,
)


def get_solver(config):
    dim_signal = config.geometry.dim_signal

    if dim_signal == 1:

        solver = NeuralEikonalSolver_1D(
            dim_signal=dim_signal,
            backbone_type=config.solver.backbone_type,
            hidden_dim=config.solver.hidden_dim,
            num_layers=config.solver.num_layers,
            original_means=config.geometry.gmm_means,
            original_variance=config.geometry.gmm_variances,
            weights=config.geometry.gmm_weights,
            activation=config.solver.activation,
            nonlinearity=config.solver.nonlinearity,
            use_fourier_features=config.solver.use_fourier_features,
            fourier_embed_scale=config.solver.fourier_embed_scale,
            fourier_embed_dim=config.solver.fourier_embed_dim,
            use_reparam=config.solver.use_reparam,
            reparam_mean=config.solver.reparam_mean,
            reparam_std=config.solver.reparam_std,
            xmin=config.geometry.x_min,
            xmax=config.geometry.x_max,
            factored=config.eikonal.factored,
            normalize_domain=config.solver.normalized,
            lambda_min=config.geometry.lambda_min,
            lambda_max=config.geometry.lambda_max,
        )
    else:
        raise NotImplementedError(
            f"Eikonal solver for dim_signal={dim_signal} not implemented."
        )
    return solver
