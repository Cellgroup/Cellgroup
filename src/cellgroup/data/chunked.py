import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class ChunkedDataset(Dataset):
    """A PyTorch Dataset that loads the data on-the-fly."""

    def __init__(self, data_dir: Path):
        """Initialize the dataset.

        Parameters
        ----------
        data_dir : Path
            The directory containing the data.
        """
        self.data_dir = data_dir

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        return np.load(self.data_dir / f"{idx}.npy")