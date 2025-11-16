import torch
from torch.utils.data import Dataset
import numpy as np

class DataSet_hyper(Dataset):
    """Dataset for head shape prediction (dx/dy coordinates)"""
    def __init__(self, data_list, feature_stats_path, global_stats_path):
        self.data = data_list
        self.feature_stats = self._load_stats(feature_stats_path)
        self.global_stats = self._load_stats(global_stats_path)

    def _load_stats(self, path):
        """Load statistics (mean, std, max, min) from text file"""
        stats = {}
        with open(path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                stats[key] = np.array([float(x) for x in value.split(',')])
        return stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data structure: [features, dx, dy]
        item = self.data[idx]
        features = item[:-2].astype(np.float32)  # Input features
        dx = item[-2].astype(np.float32)         # dx coordinates (21,51)
        dy = item[-1].astype(np.float32)         # dy coordinates (21,51)

        # Normalize features
        feat_mean = self.feature_stats['mean']
        feat_std = self.feature_stats['std'] + 1e-6  # Avoid division by zero
        features = (features - feat_mean) / feat_std

        # Normalize dx/dy to [-1, 1]
        dx_max, dx_min = self.global_stats['dx_max'], self.global_stats['dx_min']
        dy_max, dy_min = self.global_stats['dy_max'], self.global_stats['dy_min']
        dx_norm = 2 * (dx - dx_min) / (dx_max - dx_min + 1e-6) - 1
        dy_norm = 2 * (dy - dy_min) / (dy_max - dy_min + 1e-6) - 1

        # Combine dx and dy into (2, 21, 51) tensor
        target = np.stack([dx_norm, dy_norm], axis=0)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(dx_max, dtype=torch.float32),
            torch.tensor(dx_min, dtype=torch.float32),
            torch.tensor(dy_max, dtype=torch.float32),
            torch.tensor(dy_min, dtype=torch.float32)
        )
