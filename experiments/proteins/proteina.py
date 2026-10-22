"""Proteina as a deterministic ODE map from a Gaussian source to backbones."""

import contextlib
import io
import os

import torch

from experiments.common import REPO_DIR, add_prior_to_path

PROTEINA_DIR = add_prior_to_path("proteina")
# Proteina reads auxiliary files (e.g. the CATH label mapping) from $DATA_PATH
os.environ.setdefault("DATA_PATH", str(REPO_DIR / "data" / "proteina"))

DEFAULT_CKPT = str(REPO_DIR / "pretrained" / "proteina_v1.3_DFS_60M_notri.ckpt")
ODE_DT = 0.02  # 50 ODE steps


def load_model(ckpt, device):
    from proteinfoundation.proteinflow.proteina import Proteina

    return Proteina.load_from_checkpoint(ckpt, map_location="cpu").to(device).eval()


@torch.no_grad()
def generate(model, nsamples, n_residues, source=None):
    """Sample backbones with the probability flow ODE (50 steps, no guidance).

    Args:
        source: optional standard Gaussian source ``[nsamples, n_residues, 3]``.
            Its centre of mass is removed inside the map, so that the source
            prior stays N(0, I).

    Returns:
        atom37 coordinates ``[nsamples, n_residues, 37, 3]`` in Angstrom.
    """
    original_sample_reference = model.fm.sample_reference
    if source is not None:
        centered = source - source.mean(dim=1, keepdim=True)
        model.fm.sample_reference = lambda *a, **kw: centered
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            x = model.generate(
                nsamples=nsamples,
                n=n_residues,
                dt=torch.tensor(ODE_DT, dtype=torch.float32),
                self_cond=True,
                cath_code=None,
                guidance_weight=1.0,
                autoguidance_ratio=0.0,
                dtype=torch.float32,
                schedule_mode="log",
                schedule_p=2.0,
                sampling_mode="vf",
                sc_scale_noise=0.0,
                sc_scale_score=1.0,
                gt_mode="1/t",
                gt_p=1.0,
                gt_clamp_val=None,
            )
    finally:
        model.fm.sample_reference = original_sample_reference
    return model.samples_to_atom37(x)
