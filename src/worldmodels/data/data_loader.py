# data_loader.py
# ----------------------------------------------------------
# Data loading & preprocessing for the World-Models VAE
# (CarRacing-v0, DoomTakeCover-v0, etc.)
# ----------------------------------------------------------
from __future__ import annotations

import bisect
import random
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


def _default_transform(sz: Tuple[int, int] = (64, 64)) -> Callable[[Image.Image], torch.Tensor]:
    """Resize and convert PIL image to  [0,1]  float32 tensor."""
    return transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # (C,H,W) in [0,1]
    ])


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------
class WorldModelsFrames(Dataset):
    """
    Frame dataset that reads exclusively from one or more *.npy files
    located directly in `root`.  Each file must be an (N,H,W,3) uint8 array.
    """

    def __init__(
            self,
            root: str | Path,
            transform: Optional[Callable[[Image.Image | np.ndarray], torch.Tensor]] = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        t0 = time.perf_counter()
        print(f"[WorldModelsFrames] Scanning *.npy files under “{self.root}”…",
              flush=True)

        # ---------- gather metadata only (no memmaps kept) ------------------
        self.npy_files: list[Path] = sorted(self.root.glob("*.npy"))
        if not self.npy_files:
            raise RuntimeError(f"No .npy files found in {self.root}")

        self.frame_counts: list[int] = []
        total_bytes = 0
        for f in self.npy_files:
            arr = np.load(f, mmap_mode="r")  # open once to read shape
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError(f"Unexpected shape {arr.shape} in {f}")
            self.frame_counts.append(arr.shape[0])
            total_bytes += arr.nbytes
            del arr  # drop memmap immediately

        # prefix sums for O(log M) lookup
        self.cum_lengths = np.cumsum(self.frame_counts).tolist()
        self.length = self.cum_lengths[-1]

        dt = time.perf_counter() - t0
        size_gb = total_bytes / (1024 ** 3)
        print(f"[WorldModelsFrames] Indexed {len(self.npy_files)} file(s) – "
              f"{self.length:,} frames, ≈{size_gb:.2f} GB "
              f"in {dt:.2f}s", flush=True)

        # each process keeps its own memmap cache {file_idx: memmap}
        self._arrays: dict[int, np.memmap] = {}

        self.transform = transform or _default_transform()

    # ---------- make the object picklable ----------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}  # never pickle open memmaps
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._arrays = {}  # each worker gets a fresh cache

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def _ensure_open(self, file_idx: int) -> np.memmap:
        arr = self._arrays.get(file_idx)
        if arr is None:
            fname = self.npy_files[file_idx].name
            arr = np.load(self.npy_files[file_idx], mmap_mode="r")
            self._arrays[file_idx] = arr
        return arr

    def _load_pil(self, idx: int) -> Image.Image:
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev = self.cum_lengths[file_idx - 1] if file_idx else 0
        local_idx = idx - prev
        arr = self._ensure_open(file_idx)
        return Image.fromarray(arr[local_idx])

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._load_pil(idx)
        return self.transform(img)


# ----------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------
def make_vae_dataloaders(
        root: str | Path,
        *,
        batch_size: int = 64,
        num_workers: int = 4,
        train_split: float = 0.9,
        shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader)
    """
    full_ds = WorldModelsFrames(root)
    N = len(full_ds)
    idxs = list(range(N))
    random.shuffle(idxs)

    split = int(N * train_split)
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    train_ds = Subset(full_ds, train_idxs)
    val_ds = Subset(full_ds, val_idxs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[make_dataloaders] train={len(train_ds):,}  "
          f"val={len(val_ds):,}  workers={num_workers}", flush=True)

    return train_loader, val_loader
