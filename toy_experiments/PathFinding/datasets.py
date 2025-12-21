import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class CoordinateDataset(Dataset):
    def __init__(
        self,
        num_pairs,
        dim_signal,
        precomputed_dir,
        sampling_strategy="uniform",
        x_min=None,
        x_max=None,
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


        # Coordinate bounds: `time` component, and followed by the `space` components coords
        self.x_min = x_min
        self.x_max = x_max

        assert len(self.x_min) == (self.dim_signal+1)and len(self.x_max) == (self.dim_signal+1)

        # Metadata file
        self.metadata_path = os.path.join(precomputed_dir, "metadata.json")
        os.makedirs(precomputed_dir, exist_ok=True)

        # Precompute or validate precomputed data
        self.data = None
        self._prepare_data()

    def _prepare_data(self):
        """
        Ensures precomputed data is consistent with metadata. Recomputes if necessary.
        """
        metadata = self._load_metadata()

        # Check if recomputation is needed

        saved_file_exist =  os.path.exists(os.path.join(self.precomputed_dir, f"samples.pt"))

        current_params = {
            "num_pairs": self.num_pairs,
            "dim_signal": self.dim_signal,
            "x_min": self.x_min.tolist() if self.x_min is not None else None,
            "x_max": self.x_max.tolist() if self.x_min is not None else None,
            "sampling_strategy": self.sampling_strategy,
        }

        need_recompute = (
            self.force_recompute or metadata == None or metadata != current_params or not saved_file_exist
        )

        if need_recompute:
            self._clear_precomputed_data()
            self._save_metadata(current_params)
            self._generate_coords()

        # Load precomputed data into memory if save_data is True
        if self.save_data:
            precomputed_file = os.path.join(self.precomputed_dir, f"samples.pt")
            self.data = torch.load(precomputed_file, weights_only=False)


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


    def _generate_coords(self):
        
        coords = None
        if self.sampling_strategy == "uniform":
            """Generate random coordinate chunks for a single sample."""
            coords = np.random.rand(self.num_pairs * 2, self.dim_signal+1)
            coords = (self.x_max - self.x_min) * coords + self.x_min
            np.random.shuffle(coords)
            coords=torch.tensor(coords, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Sampling strategy '{self.sampling_strategy}' is not implemented.")
        
        
        self.data = coords.view(self.num_pairs, 2, self.dim_signal+1)

        # Save precomputed data to disk
        precomputed_file = os.path.join(self.precomputed_dir, f"samples.pt")
        torch.save(coords, precomputed_file)
        

    def __len__(self):
        """Return the total number of sample pairs."""
        return len(self.data) if self.data is not None else 0
    
    def __getitem__(self, idx):
        return self.data[idx], np.array([])




