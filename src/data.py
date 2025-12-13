from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TokenNPYDataset(Dataset):
    """
    Loads .npy token sequences from manifest.parquet.

    We reserve token id 0 for PAD in batching.
    Since tokenization emits ids starting at 0, we SHIFT ALL TOKENS by +1 here.
    """
    def __init__(
        self,
        manifest_path: str,
        split: str,
        seq_len: int = 1024,
        random_crop: bool = True,
        seed: int = 42,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.rng = np.random.default_rng(seed)

        m = pd.read_parquet(self.manifest_path)
        m = m[m["split"] == split].copy()
        m = m[m["status"].isin(["ok", "skipped_existing"])].copy()
        m = m[m["token_path"].astype(str).str.len() > 0].reset_index(drop=True)

        if len(m) == 0:
            raise RuntimeError(f"No samples found for split={split} in {manifest_path}")

        self.m = m

    def __len__(self) -> int:
        return len(self.m)

    def _select_window(self, arr: np.ndarray) -> np.ndarray:
        L = len(arr)
        if L >= self.seq_len:
            if self.random_crop:
                start = int(self.rng.integers(0, L - self.seq_len + 1))
            else:
                start = 0
            return arr[start : start + self.seq_len]
        out = np.zeros((self.seq_len,), dtype=np.int64)  # PAD=0
        out[:L] = arr
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.m.iloc[idx]
        arr = np.load(row["token_path"]).astype(np.int64)

        # shift: reserve PAD=0
        arr = arr + 1

        x = self._select_window(arr)

        # next token prediction
        input_ids = torch.tensor(x[:-1], dtype=torch.long)
        labels = torch.tensor(x[1:], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}
