"""GeoLDM (GEOM-Drugs) as a deterministic map from a Gaussian source to molecules."""

import os
import pickle
import sys
from pathlib import Path

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import QED, RDConfig

from experiments.common import add_prior_to_path

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

add_prior_to_path("GeoLDM")
N_DIMS = 3


def load_model(model_path, device):
    """Load a GeoLDM checkpoint directory (args.pickle + generative_model*.npy)."""
    from configs.datasets_config import get_dataset_info
    from qm9.models import get_latent_diffusion

    with open(Path(model_path) / "args.pickle", "rb") as f:
        train_args = pickle.load(f)
    # GeoLDM picks the device of the autoencoder from the training arguments
    train_args.cuda = torch.device(device).type == "cuda"

    dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h)
    model, _, _ = get_latent_diffusion(train_args, device, dataset_info, dataloader_train=None)
    model.to(device)

    fn = "generative_model_ema.npy" if train_args.ema_decay > 0 else "generative_model.npy"
    model.load_state_dict(torch.load(Path(model_path) / fn, map_location=device))
    model.eval()
    return model, dataset_info


def build_masks(batch_size, n_nodes, max_n_nodes, device):
    node_mask = torch.zeros(batch_size, max_n_nodes, 1, device=device)
    node_mask[:, :n_nodes, :] = 1.0

    edge_mask = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)
    diag = ~torch.eye(max_n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    edge_mask = (edge_mask * diag).view(batch_size * max_n_nodes * max_n_nodes, 1)
    return node_mask, edge_mask


def source_dim(model, n_nodes):
    return n_nodes * (N_DIMS + model.in_node_nf)


def sample_source(model, batch_size, n_nodes, device):
    """Standard Gaussian source over the active atoms, flattened.

    A plain standard Gaussian is used for all dimensions, including the
    positions, rather than the zero centre-of-mass Gaussian of the diffusion
    model. The centre of mass is projected out after the first DDIM step, so
    only 3 dimensions are redundant.
    """
    return torch.randn(batch_size, source_dim(model, n_nodes), device=device)


@torch.no_grad()
def decode(z_flat, model, n_nodes, max_n_nodes, ddim_steps):
    """Deterministic DDIM decoding of flat source samples.

    Returns positions ``[B, max_n_nodes, 3]`` and one-hot atom types
    ``[B, max_n_nodes, num_classes]``.
    """
    batch_size = z_flat.shape[0]
    feat_dim = N_DIMS + model.in_node_nf
    z_active = z_flat.view(batch_size, n_nodes, feat_dim)
    pad = torch.zeros(batch_size, max_n_nodes - n_nodes, feat_dim, device=z_flat.device)
    z_T = torch.cat([z_active, pad], dim=1)

    node_mask, edge_mask = build_masks(batch_size, n_nodes, max_n_nodes, z_flat.device)
    x, h = model.sample_ddim(z_T, node_mask, edge_mask, None, num_steps=ddim_steps)
    return x, h["categorical"]


def build_molecule(positions, atom_types, dataset_info):
    """RDKit molecule of one generated sample, or None if it is invalid or disconnected."""
    from qm9 import rdkit_functions

    try:
        mol = rdkit_functions.build_molecule(positions, atom_types, dataset_info)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        Chem.FastFindRings(mol)
        return mol
    except Exception:
        return None


def score_molecules(positions, one_hot, n_atoms, dataset_info):
    """(QED, SA) of each molecule; (None, None) for invalid or disconnected ones."""
    results = []
    for i in range(positions.shape[0]):
        mol = build_molecule(
            positions[i, :n_atoms].cpu(), one_hot[i, :n_atoms].cpu().argmax(dim=1), dataset_info
        )
        if mol is None:
            results.append((None, None))
            continue
        try:
            with rdBase.BlockLogs():
                results.append((QED.qed(mol), sascorer.calculateScore(mol)))
        except Exception:
            results.append((None, None))
    return results


def log_reward(qed, sa, weight):
    """log g(x): -w for disconnected molecules, otherwise w * (QED + SA_norm) / 2."""
    if qed is None:
        return -weight
    sa_norm = (10.0 - sa) / 9.0
    return weight * (qed + sa_norm) / 2.0
