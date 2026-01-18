import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class CoordinateDataset(Dataset):
    def __init__(
        self,
        num_pairs,
        dim_signal,
        precomputed_dir,
        sampling_strategy="uniform",
        mixed_uniform_fraction=0.5,
        x_min=None,
        x_max=None,
        solver=None,
        trajectory_n_trajectories=100,
        trajectory_n_steps=512,
        trajectory_n_points_per_trajectory=10,
        trajectory_t_start=1.0,
        trajectory_t_end=0.0,
        trajectory_oversample_factor=2.0,
        save_data=True,
        force_recompute=False,
    ):
        """
        CoordinateDataset generates coordinate pairs and their interpolated values
        for a given base dataset. Optionally caches precomputed data for faster access.

        Args:
            num_pairs (int): Number of coordinate pairs.
            dim_signal (int): Dimensionality of the signal space (excluding time).
            precomputed_dir (str): Directory for storing/retrieving precomputed data.
            x_min (list or None): Minimum coordinate values for each dimension.
            x_max (list or None): Maximum coordinate values for each dimension.
            save_data (bool): If True, load precomputed data into memory.
            force_recompute (bool): If True, force recomputation of precomputed data.
        """
        self.num_pairs = num_pairs

        # dim_signal is the number of space dimensions (excluding time)
        self.dim_signal = dim_signal

        self.precomputed_dir = precomputed_dir
        self.save_data = save_data
        self.force_recompute = force_recompute

        self.sampling_strategy = sampling_strategy
        self.mixed_uniform_fraction = mixed_uniform_fraction
        self.coords = None
        self.coord_weights = None
        self.sample_weights = None

        # Coordinate bounds: `time` component, and followed by the `space` components coords
        self.x_min = np.array(x_min) if x_min is not None else np.zeros(dim_signal + 1)
        self.x_max = np.array(x_max) if x_max is not None else np.ones(dim_signal + 1)

        assert len(self.x_min) == (self.dim_signal + 1) and len(self.x_max) == (
            self.dim_signal + 1
        )

        # Metadata file
        self.metadata_path = os.path.join(precomputed_dir, "metadata.json")
        os.makedirs(precomputed_dir, exist_ok=True)

        self.solver = solver
        self.trajectory_n_trajectories = trajectory_n_trajectories
        self.trajectory_n_steps = trajectory_n_steps
        self.trajectory_n_points_per_trajectory = trajectory_n_points_per_trajectory
        self.trajectory_t_start = trajectory_t_start
        self.trajectory_t_end = trajectory_t_end
        self.trajectory_oversample_factor = trajectory_oversample_factor

        # Precompute or validate precomputed data
        self.data = None
        self._prepare_data()

    def _prepare_data(self):
        """
        Ensures precomputed data is consistent with metadata. Recomputes if necessary.
        """
        metadata = self._load_metadata()

        # Check if recomputation is needed

        saved_file_exist = os.path.exists(
            os.path.join(self.precomputed_dir, f"samples.pt")
        )

        current_params = {
            "num_pairs": self.num_pairs,
            "dim_signal": self.dim_signal,
            "x_min": self.x_min.tolist() if self.x_min is not None else None,
            "x_max": self.x_max.tolist() if self.x_min is not None else None,
            "sampling_strategy": self.sampling_strategy,
            "mixed_uniform_fraction": self.mixed_uniform_fraction,
        }
        if self.sampling_strategy == "trajectory":
            current_params.update(
                {
                    "trajectory_n_trajectories": self.trajectory_n_trajectories,
                    "trajectory_n_steps": self.trajectory_n_steps,
                    "trajectory_n_points_per_trajectory": self.trajectory_n_points_per_trajectory,
                    "trajectory_t_start": self.trajectory_t_start,
                    "trajectory_t_end": self.trajectory_t_end,
                    "trajectory_oversample_factor": self.trajectory_oversample_factor,
                }
            )

        need_recompute = (
            self.force_recompute
            or metadata == None
            or metadata != current_params
            or not saved_file_exist
        )

        if need_recompute:
            self._clear_precomputed_data()
            self._save_metadata(current_params)
            self._generate_coords()

        # Load precomputed data into memory if save_data is True
        if self.save_data:
            precomputed_file = os.path.join(self.precomputed_dir, f"samples.pt")
            self.data = torch.load(precomputed_file, weights_only=False)
        if self.data is not None:
            self.coords = self.data.view(-1, self.dim_signal + 1).clone()
            self._init_sample_weights()
            self._rebuild_pairs_from_coords(shuffle=False)

    def _init_sample_weights(self):
        self.coord_weights = torch.ones(self.coords.shape[0], dtype=torch.float32)
        self.sample_weights = torch.ones(self.num_pairs, dtype=torch.float32)

    def _load_metadata(self):
        """Load metadata from disk, if it exists."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return None

    def _save_metadata(self, metadata):
        """Save metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

    def _clear_precomputed_data(self):
        """Remove all precomputed data files."""
        for file in os.scandir(self.precomputed_dir):
            if file.is_file():
                os.remove(file.path)

    def _generate_uniform_coords(self, num_points):
        coords = np.random.rand(num_points, self.dim_signal + 1)
        coords = (self.x_max - self.x_min) * coords + self.x_min
        np.random.shuffle(coords)
        return torch.tensor(coords, dtype=torch.float32)

    def _sample_coords(self, num_points):
        if num_points <= 0:
            return torch.empty((0, self.dim_signal + 1), dtype=torch.float32)
        if self.sampling_strategy == "uniform":
            return self._generate_uniform_coords(num_points)
        if self.sampling_strategy == "trajectory":
            return self._generate_trajectory_coords(num_points)
        if self.sampling_strategy == "mixed":
            uniform_points = int(round(num_points * self.mixed_uniform_fraction))
            uniform_points = min(max(uniform_points, 0), num_points)
            traj_points = num_points - uniform_points
            pieces = []
            if uniform_points > 0:
                pieces.append(self._generate_uniform_coords(uniform_points))
            if traj_points > 0:
                pieces.append(self._generate_trajectory_coords(traj_points))
            if not pieces:
                return torch.empty((0, self.dim_signal + 1), dtype=torch.float32)
            coords = torch.cat(pieces, dim=0)
            perm = torch.randperm(coords.shape[0])
            return coords[perm]
        raise NotImplementedError(
            f"Sampling strategy '{self.sampling_strategy}' is not implemented."
        )

    def _generate_coords(self):

        coords = self._sample_coords(self.num_pairs * 2)
        if coords.numel() == 0:
            raise ValueError(f"{self.sampling_strategy} sampling produced no points.")
        self.coords = coords.clone()
        self._init_sample_weights()
        self._rebuild_pairs_from_coords(shuffle=False)

        # Save precomputed data to disk
        precomputed_file = os.path.join(self.precomputed_dir, f"samples.pt")
        torch.save(self.data, precomputed_file)

    def _generate_trajectory_coords(self, target_points=None):
        if self.solver is None:
            raise ValueError("Trajectory sampling requires a solver instance.")
        assert hasattr(
            self.solver, "sample_trajectory_points"
        ), "Solver must implement sample_trajectory_points for trajectory sampling."

        if target_points is None:
            target_points = self.num_pairs * 2
        points = []
        collected = 0
        max_rounds = 50
        points_per_traj = max(1, int(self.trajectory_n_points_per_trajectory))

        with tqdm(total=target_points, desc="Sampling trajectories", unit="pts") as pbar:
            for _ in range(max_rounds):
                remaining = target_points - collected
                if remaining <= 0:
                    break
                desired = int(np.ceil(remaining * self.trajectory_oversample_factor))
                n_trajectories = max(1, int(np.ceil(desired / points_per_traj)))
                sampled = self.solver.sample_trajectory_points(
                    n_trajectories=n_trajectories,
                    n_steps=self.trajectory_n_steps,
                    n_points_per_trajectory=points_per_traj,
                    x_range=(self.x_min[0], self.x_max[0]),
                    t_start=self.trajectory_t_start,
                    t_end=self.trajectory_t_end,
                )
                if not torch.is_tensor(sampled):
                    sampled = torch.tensor(sampled, dtype=torch.float32)
                else:
                    sampled = sampled.to(dtype=torch.float32)

                if sampled.numel() == 0:
                    continue

                mask = torch.ones(sampled.shape[0], dtype=torch.bool)
                for dim in range(self.dim_signal + 1):
                    mask &= sampled[:, dim] >= self.x_min[dim]
                    mask &= sampled[:, dim] <= self.x_max[dim]
                sampled = sampled[mask]

                if sampled.numel() == 0:
                    continue

                points.append(sampled)
                collected += sampled.shape[0]
                pbar.update(sampled.shape[0])

        if collected < target_points:
            raise ValueError(
                f"Trajectory sampling produced {collected} points, "
                f"but {target_points} are required."
            )

        all_points = torch.cat(points, dim=0)
        perm = torch.randperm(all_points.shape[0])[:target_points]
        coords = all_points[perm]
        return coords

    def __len__(self):
        """Return the total number of sample pairs."""
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        return self.data[idx], np.array([])

    def _rebuild_pairs_from_coords(self, shuffle=False):
        if self.coords is None:
            raise ValueError("Pair rebuild requires coordinate storage.")
        if shuffle:
            perm = torch.randperm(self.coords.shape[0])
            self.coords = self.coords[perm]
            if self.coord_weights is not None:
                self.coord_weights = self.coord_weights[perm]
        self.data = self.coords.view(self.num_pairs, 2, self.dim_signal + 1)
        if self.coord_weights is not None:
            pair_weights = self.coord_weights.view(self.num_pairs, 2).mean(dim=1)
            if self.sample_weights is None or self.sample_weights.shape != pair_weights.shape:
                self.sample_weights = pair_weights.clone()
            else:
                self.sample_weights.copy_(pair_weights)

    def reshuffle_pairs(self):
        self._rebuild_pairs_from_coords(shuffle=True)

    def update_coordinate_weights(self, weights, min_weight=0.1, max_weight=0.9):
        if self.coords is None:
            raise ValueError("update_coordinate_weights requires in-memory dataset data.")
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.shape[0] != self.coords.shape[0]:
            raise ValueError("Weights must match coordinate count.")
        max_val = weights.max().clamp_min(1e-12)
        weights = weights / max_val
        weights = weights.clamp(min=min_weight, max=max_weight)
        if self.coord_weights is None or self.coord_weights.shape != weights.shape:
            self.coord_weights = weights.clone()
        else:
            self.coord_weights.copy_(weights)
        self._rebuild_pairs_from_coords(shuffle=False)


def get_dataloaders(config, solver=None):

    dim_signal = config.geometry.dim_signal
    num_pairs = config.data.num_pairs
    x_min = config.geometry.x_min
    x_max = config.geometry.x_max
    sampling_strategy = config.data.sampling_strategy
    mixed_uniform_fraction = float(
        getattr(config.data, "mixed_uniform_fraction", 0.5)
    )
    num_workers = config.data.num_workers
    if isinstance(num_workers, str) and num_workers.lower() in {"max", "auto"}:
        num_workers = os.cpu_count() or 0
    elif isinstance(num_workers, int) and num_workers < 0:
        num_workers = os.cpu_count() or 0
    traj_n_trajectories = getattr(config.data, "trajectory_n_trajectories", 100)
    traj_n_steps = getattr(config.data, "trajectory_n_steps", 512)
    traj_points_per = getattr(config.data, "trajectory_n_points_per_trajectory", 10)
    traj_t_start = getattr(config.data, "trajectory_t_start", 1.0)
    traj_t_end = getattr(config.data, "trajectory_t_end", 0.0)
    traj_oversample = getattr(config.data, "trajectory_oversample_factor", 2.0)

    suffix = ""
    if sampling_strategy == "mixed":
        suffix = f"_u{int(round(mixed_uniform_fraction * 100))}"

    train_dataset = CoordinateDataset(
        num_pairs=num_pairs,
        dim_signal=dim_signal,
        precomputed_dir=os.path.join(
            "toy_experiments",
            "PathFinding",
            "data",
            f"{sampling_strategy}{suffix}__train",
        ),
        sampling_strategy=sampling_strategy,
        mixed_uniform_fraction=mixed_uniform_fraction,
        x_min=x_min,
        x_max=x_max,
        solver=solver,
        trajectory_n_trajectories=traj_n_trajectories,
        trajectory_n_steps=traj_n_steps,
        trajectory_n_points_per_trajectory=traj_points_per,
        trajectory_t_start=traj_t_start,
        trajectory_t_end=traj_t_end,
        trajectory_oversample_factor=traj_oversample,
        save_data=config.data.train_save_data,
        force_recompute=config.data.train_force_recompute,
    )

    val_dataset = CoordinateDataset(
        num_pairs=num_pairs,
        dim_signal=dim_signal,
        precomputed_dir=os.path.join(
            "toy_experiments",
            "PathFinding",
            "data",
            f"{sampling_strategy}{suffix}__val",
        ),
        sampling_strategy=sampling_strategy,
        mixed_uniform_fraction=mixed_uniform_fraction,
        x_min=x_min,
        x_max=x_max,
        solver=solver,
        trajectory_n_trajectories=traj_n_trajectories,
        trajectory_n_steps=traj_n_steps,
        trajectory_n_points_per_trajectory=traj_points_per,
        trajectory_t_start=traj_t_start,
        trajectory_t_end=traj_t_end,
        trajectory_oversample_factor=traj_oversample,
        save_data=config.data.val_save_data,
        force_recompute=config.data.val_force_recompute,
    )

    test_dataset = CoordinateDataset(
        num_pairs=num_pairs,
        dim_signal=dim_signal,
        precomputed_dir=os.path.join(
            "toy_experiments",
            "PathFinding",
            "data",
            f"{sampling_strategy}{suffix}__test",
        ),
        sampling_strategy=sampling_strategy,
        mixed_uniform_fraction=mixed_uniform_fraction,
        x_min=x_min,
        x_max=x_max,
        solver=solver,
        trajectory_n_trajectories=traj_n_trajectories,
        trajectory_n_steps=traj_n_steps,
        trajectory_n_points_per_trajectory=traj_points_per,
        trajectory_t_start=traj_t_start,
        trajectory_t_end=traj_t_end,
        trajectory_oversample_factor=traj_oversample,
        save_data=config.data.test_save_data,
        force_recompute=config.data.test_force_recompute,
    )

    weighted_sampling_enabled = bool(
        getattr(config.data, "weighted_sampling_enabled", False)
    )
    train_sampler = None
    train_shuffle = True
    if weighted_sampling_enabled:
        from torch.utils.data import WeightedRandomSampler

        train_sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
    )

    return train_loader, val_loader, test_loader
