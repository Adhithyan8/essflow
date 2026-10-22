"""Designability and diversity of generated protein backbones.

Designability is the percentage of samples with scRMSD < 2 Angstrom. Diversity
is the mean pairwise TM-score among the designable samples (lower is more
diverse), computed with Foldseek's all-vs-all TM-align search.

Example:
    python -m experiments.proteins.evaluate results/proteins/*.pt
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

CA_IDX = 1  # index of the C-alpha atom in atom37


def write_ca_pdb(path, ca):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(ca.tolist()):
            f.write(
                f"ATOM  {i+1:5d}  CA  GLY A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


def mean_pairwise_tm_score(ca_list):
    tmp = Path(tempfile.mkdtemp(prefix="foldseek_"))
    pdb_dir = tmp / "pdbs"
    pdb_dir.mkdir()
    try:
        for i, ca in enumerate(ca_list):
            write_ca_pdb(pdb_dir / f"s{i:05d}.pdb", ca)

        search_out = tmp / "search"
        subprocess.run(
            [
                "foldseek", "easy-search", str(pdb_dir), str(pdb_dir), str(search_out), str(tmp),
                "--alignment-type", "1", "--cov-mode", "0", "--min-seq-id", "0",
                "--exhaustive-search", "1", "--format-output", "query,target,qtmscore",
                "-e", "inf",
            ],
            check=True,
            capture_output=True,
        )
        tm_scores = []
        with open(search_out) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[0] != parts[1]:
                    tm_scores.append(float(parts[2]))
        return sum(tm_scores) / len(tm_scores) if tm_scores else float("nan")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(atom37, scrmsd):
    designable = scrmsd < 2.0
    n_des = int(designable.sum())
    tm_score = float("nan")
    if n_des >= 2:
        ca = [atom37[i, :, CA_IDX, :] for i in designable.nonzero().squeeze(-1).tolist()]
        tm_score = mean_pairwise_tm_score(ca)
    return {
        "n_total": len(scrmsd),
        "scrmsd": scrmsd.mean().item(),
        "designability": 100 * n_des / len(scrmsd),
        "tm_score": tm_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated protein backbones")
    parser.add_argument("files", type=str, nargs="+")
    args = parser.parse_args()

    header = f"{'File':40s} {'N':>5s} {'scRMSD':>7s} {'Des. %':>7s} {'TM-score':>9s}"
    rows = []
    for f in args.files:
        data = torch.load(f, map_location="cpu")
        r = evaluate(data["atom37"], data["scrmsd"])
        rows.append(
            f"{f[-40:]:40s} {r['n_total']:5d} {r['scrmsd']:7.2f} "
            f"{r['designability']:7.1f} {r['tm_score']:9.3f}"
        )
    print(header)
    print("-" * len(header))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
