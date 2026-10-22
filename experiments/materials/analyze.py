"""Property statistics and histograms of generated materials.

Expects the outputs of each method in ``<results-dir>/<task>/<method>*.pt``,
e.g. ``results/materials/bulk_modulus/essflow_1.pt`` ... ``essflow_8.pt``.
Prints the mean absolute error to the target (mean energy above hull for
e_hull, match rate for space_group) and plots the property histograms.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.materials.tasks import TASKS

METHODS = {
    "unconditional": "Unconditional",
    "dflow": "D-Flow",
    "pnpflow": "PnP-Flow",
    "daps": "DAPS",
    "essflow": "ESS-Flow",
    "reweighted": "Reweighted",
}
TITLES = {
    "bulk_modulus": "Bulk modulus: 300 GPa",
    "shear_modulus": "Shear modulus: 200 GPa",
    "band_gap": "Band gap: 10 eV",
    "e_hull": "Energy above hull <= 0 eV",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Property statistics and histograms")
    parser.add_argument("--task", type=str, choices=list(TASKS), required=True)
    parser.add_argument("--results-dir", type=str, default="results/materials")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["dflow", "pnpflow", "daps", "essflow"],
                        choices=list(METHODS), help="methods to plot")
    parser.add_argument("--colors", type=str, nargs="+",
                        default=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--figure-size", type=float, nargs=2, default=[5.5, 1.2])
    parser.add_argument("--output", type=str, default=None,
                        help="figure path (default: <results-dir>/<task>/histogram.pdf)")
    return parser.parse_args()


def load_method(task_dir, method):
    files = sorted(
        task_dir.glob(f"{method}*.pt"),
        key=lambda p: (len(p.stem), p.stem),
    )
    files = [f for f in files if not f.stem.endswith("_weights")]
    if not files:
        return None, None
    data = [torch.load(f, map_location="cpu") for f in files]
    if "space_groups" in data[0]:
        values = sum((list(d["space_groups"]) for d in data), [])
    else:
        values = torch.cat([d["property"].reshape(-1) for d in data]).numpy()
    nfe = torch.cat([d["nfe"] for d in data]) if "nfe" in data[0] else None
    print(f"Loaded {METHODS[method]}: {len(values)} samples from {len(files)} file(s)")
    return values, nfe


def setup_matplotlib_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 9,
        }
    )


def plot_histograms(values, target, methods, colors, args, output):
    setup_matplotlib_style()
    all_values = np.concatenate([values[m] for m in values])
    bins = np.linspace(all_values.min(), all_values.max(), args.num_bins + 1)

    fig, axes = plt.subplots(1, len(methods), figsize=args.figure_size, sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, method, color) in enumerate(zip(axes, methods, colors)):
        ax.hist(values["unconditional"], bins=bins, density=True, histtype="step",
                linewidth=1, color="grey", alpha=0.8, label="Unconditional")
        ax.hist(values[method], bins=bins, density=True, color=color, alpha=0.8,
                label=METHODS[method])
        ax.axvline(x=target, color="black", linestyle="--", linewidth=1, label="Target")
        if i == 0:
            ax.set_ylabel("Density")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
        ax.set_title(METHODS[method])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output}")


def main():
    args = parse_args()
    task_dir = Path(args.results_dir) / args.task
    target = TASKS[args.task]["target"]

    values, essflow_nfe = {}, None
    for method in METHODS:
        v, nfe = load_method(task_dir, method)
        if v is not None:
            values[method] = v
        if method == "essflow":
            essflow_nfe = nfe

    if args.task == "space_group":
        print(f"\nSpace group match rate (target {target}):")
        print("-" * 60)
        for method, v in values.items():
            matches = sum(sg == target for sg in v)
            print(f"{METHODS[method]:15s}: {100 * matches / len(v):5.1f}% ({matches}/{len(v)})")
    else:
        if args.task == "e_hull":
            print(f"\nMean ± std of the predicted energy above hull (eV/atom):")
            stats = {m: (np.mean(v), np.std(v)) for m, v in values.items()}
        else:
            print(f"\nMean absolute error ± std, {TITLES[args.task]}:")
            stats = {
                m: (np.mean(np.abs(v - target)), np.std(np.abs(v - target)))
                for m, v in values.items()
            }
        print("-" * 60)
        for method, (mean, std) in stats.items():
            print(f"{METHODS[method]:15s}: {mean:7.3f} ± {std:6.3f}")

    if essflow_nfe is not None:
        nfe = essflow_nfe.float()
        print(f"\nESS-Flow NFE per chain: mean {nfe.mean():.1f}, std {nfe.std(unbiased=False):.1f}, "
              f"min {nfe.min():.0f}, max {nfe.max():.0f}")

    methods = [m for m in args.methods if m in values]
    if args.task != "space_group" and "unconditional" in values and methods:
        colors = [args.colors[args.methods.index(m) % len(args.colors)] for m in methods]
        output = Path(args.output) if args.output else task_dir / "histogram.pdf"
        plot_histograms(values, 0.0 if target is None else target, methods, colors, args, output)


if __name__ == "__main__":
    main()
