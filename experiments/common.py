"""Small utilities shared by all experiment scripts."""

import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
PRIORS_DIR = REPO_DIR / "priors"


def add_prior_to_path(name):
    """Make a prior submodule (e.g. ``GeoLDM``) importable."""
    path = str(PRIORS_DIR / name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return Path(path)


def setup_logging(verbose=True):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed):
    """Seed python, numpy and torch. ``None`` leaves the RNGs unseeded."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    logging.info(f"Saved to {path}")


class GPUMemoryTracker:
    """Records peak GPU memory at named stages."""

    def __init__(self, device):
        self.device = device
        self.is_cuda = torch.device(device).type == "cuda"
        self.memory_stats = defaultdict(lambda: {"peak": 0, "allocated": 0})
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

    def record(self, stage_name):
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            self.memory_stats[stage_name]["allocated"] = allocated
            self.memory_stats[stage_name]["peak"] = peak

    def peak(self):
        if self.is_cuda:
            return torch.cuda.max_memory_allocated(self.device) / (1024**3)
        return 0.0

    def print_summary(self):
        if not self.is_cuda:
            logging.info("GPU memory tracking not available (not using CUDA)")
            return
        logging.info("=" * 70)
        logging.info("GPU Memory Usage Summary")
        logging.info("=" * 70)
        for stage, stats in self.memory_stats.items():
            logging.info(
                f"{stage:30s} - Peak: {stats['peak']:6.2f} GB, "
                f"Allocated: {stats['allocated']:6.2f} GB"
            )
        logging.info(f"{'Overall Peak Memory':30s} - {self.peak():6.2f} GB")
        logging.info("=" * 70)
