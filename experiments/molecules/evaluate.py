"""Metrics for generated molecules.

Conn.: percentage of connected molecules. Disconnected molecules are removed
before computing the remaining metrics. Div.: mean pairwise Tanimoto distance
between Morgan fingerprints. Uniq.: percentage of distinct canonical SMILES.
QED and SA: mean ± std. Qual.: percentage of all samples that are connected,
unique (each distinct SMILES counts once), with QED >= 0.6 and SA <= 4.

Example:
    python -m experiments.molecules.evaluate results/molecules/*.pt
"""

import argparse
import statistics

import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from experiments.molecules.geoldm import build_molecule

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def molecule_metrics(data, dataset_info):
    positions, one_hot = data["positions"], data["one_hot"]
    qeds, sas = data["qed"], data["sa"]
    n_atoms = data["num_atoms"]
    n_total = one_hot.shape[0]

    smiles, fps = [], []
    for i in range(n_total):
        mol = build_molecule(
            positions[i, :n_atoms], one_hot[i, :n_atoms].argmax(dim=1), dataset_info
        )
        smi = None
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                fps.append(_morgan_gen.GetFingerprint(mol))
            except Exception:
                pass
        smiles.append(smi)

    valid_smiles = [s for s in smiles if s is not None]
    qed_vals = [q for q in qeds if q is not None]
    sa_vals = [s for s in sas if s is not None]

    total_dist, n_pairs = 0.0, 0
    for j in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[j], fps[j + 1 :])
        total_dist += sum(1.0 - s for s in sims)
        n_pairs += len(sims)

    quality = {
        smi
        for smi, q, s in zip(smiles, qeds, sas)
        if smi is not None and q is not None and q >= 0.6 and s <= 4.0
    }

    def mean_std(values):
        if len(values) < 2:
            return float("nan"), float("nan")
        return statistics.mean(values), statistics.stdev(values)

    return {
        "n_total": n_total,
        "connected": 100 * len(qed_vals) / n_total,
        "diversity": total_dist / n_pairs if n_pairs else float("nan"),
        "unique": 100 * len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else float("nan"),
        "qed": mean_std(qed_vals),
        "sa": mean_std(sa_vals),
        "quality": 100 * len(quality) / n_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("files", type=str, nargs="+")
    args = parser.parse_args()

    from configs.datasets_config import get_dataset_info

    dataset_info = get_dataset_info("geom", remove_h=False)

    header = f"{'File':40s} {'N':>5s} {'Conn.':>6s} {'Div.':>6s} {'Uniq.':>6s} " \
             f"{'QED':>14s} {'SA':>14s} {'Qual.':>6s}"
    rows = []
    for f in args.files:
        m = molecule_metrics(torch.load(f, map_location="cpu"), dataset_info)
        rows.append(
            f"{f[-40:]:40s} {m['n_total']:5d} {m['connected']:6.1f} {m['diversity']:6.3f} "
            f"{m['unique']:6.1f} {m['qed'][0]:6.3f} ± {m['qed'][1]:5.3f} "
            f"{m['sa'][0]:6.3f} ± {m['sa'][1]:5.3f} {m['quality']:6.1f}"
        )
    print(header)
    print("-" * len(header))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
