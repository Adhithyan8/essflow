"""S.U.N.T. rate (stability, uniqueness, novelty, threshold) of generated materials.

Valid structures (volume, atom types, SMACT charge balance, interatomic
distances) are relaxed with CHGNet, and their energy above the Materials
Project convex hull (2023-02-07, from Matbench Discovery) determines stability.
Uniqueness and novelty are checked with pymatgen's StructureMatcher, the latter
against MP-20 training structures with the same set of elements. The threshold
criterion checks the predicted (ALIGNN) property or the space group.
All rates are fractions of the total number of samples, including invalid ones.
"""

import argparse
import json
import logging
import os
import pickle
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chgnet.model import StructOptimizer
from chgnet.model.model import CHGNet

# importing diffcsp.eval_utils changes the working directory to FlowMM's root
_cwd = os.getcwd()
from diffcsp.eval_utils import smact_validity, structure_validity  # noqa: E402
from diffcsp.script_utils import chemical_symbols  # noqa: E402

os.chdir(_cwd)

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core.lattice import Lattice  # noqa: E402
from pymatgen.core.structure import Structure  # noqa: E402
from pymatgen.io.cif import CifParser  # noqa: E402

from experiments.common import PRIORS_DIR, REPO_DIR, setup_logging
from experiments.materials.tasks import TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="S.U.N.T. rate of generated materials")
    parser.add_argument("--samples", type=str, nargs="+", required=True,
                        help="output files of one method (concatenated)")
    parser.add_argument("--task", type=str, choices=list(TASKS), required=True)
    parser.add_argument("--train-csv", type=str,
                        default=str(PRIORS_DIR / "flowmm/data/mp_20/train.csv"))
    parser.add_argument("--ppd-path", type=str,
                        default=str(REPO_DIR / "data/2023-02-07-ppd-mp.pkl"))
    parser.add_argument("--stability-threshold", type=float, default=0.1)
    parser.add_argument("--relaxation-steps", type=int, default=500)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON file for the results")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def load_samples(files, use_space_group):
    data = [torch.load(f) for f in files]
    samples = {
        k: torch.cat([d[k] for d in data])
        for k in ["atom_types", "frac_coords", "lattices", "lengths", "angles"]
    }
    if use_space_group:
        samples["space_groups"] = sum((list(d["space_groups"]) for d in data), [])
    else:
        samples["property"] = torch.cat([d["property"].reshape(-1) for d in data])
    return samples


def get_composition(atom_types):
    elem_counter = Counter(atom_types)
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    return elems, tuple(counts.astype("int").tolist())


def is_valid(atom_types, frac_coords, lengths, angles, lattice):
    atom_types = np.array(atom_types).astype(int)
    if (lengths < 0).any():
        return False, "non_positive_lattice"
    if np.isnan(lengths).any() or np.isnan(angles).any() or np.isnan(frac_coords).any():
        return False, "nan_value"
    if (1 > atom_types).any() or (atom_types > 104).any():
        return False, "atom_types_out_of_range"
    try:
        elements = [chemical_symbols[int(z)] for z in atom_types]
        structure = Structure(Lattice(lattice), elements, frac_coords)
        if structure.volume < 0.1:
            return False, "unrealistically_small_lattice"
        elems, comps = get_composition(atom_types)
        if not smact_validity(elems, comps):
            return False, "invalid_composition"
        if not structure_validity(structure):
            return False, "invalid_structure"
        return True, None
    except Exception as e:
        return False, f"construction_failed: {e}"


def to_structure(atom_types, frac_coords, lattice):
    elements = [chemical_symbols[int(z)] for z in atom_types.numpy()]
    return Structure(Lattice(lattice.numpy()), elements, frac_coords.numpy())


def energy_above_hull(structure, ppd_mp, relaxer, relaxation_steps):
    try:
        relaxation = relaxer.relax(structure, steps=relaxation_steps, verbose=False)
        energy_relax = relaxation["trajectory"].energies[-1]
        e_hull_per_atom = ppd_mp.get_hull_energy_per_atom(structure.composition)
        return energy_relax / len(structure) - e_hull_per_atom
    except Exception:
        return float("inf")


def element_set(structure):
    return tuple(sorted(structure.composition.elements, key=lambda x: str(x)))


def load_train_structures(train_csv):
    train_df = pd.read_csv(train_csv)
    by_elements = defaultdict(list)
    for cif in train_df["cif"]:
        try:
            structure = CifParser(StringIO(cif)).get_structures()[0]
            by_elements[element_set(structure)].append(structure)
        except Exception:
            continue
    logging.info(
        f"Loaded {sum(len(v) for v in by_elements.values())} training structures"
    )
    return by_elements


def uniqueness_and_novelty(structures, train_by_elements):
    """Returns masks of unique structures and of unique *and* novel structures."""
    matcher = StructureMatcher()
    n = len(structures)

    unique = np.ones(n, dtype=bool)
    for i in range(n):
        if not unique[i]:
            continue
        for j in range(i + 1, n):
            if unique[j]:
                try:
                    if matcher.fit(structures[i], structures[j]):
                        unique[j] = False
                except Exception:
                    continue

    novel = unique.copy()
    for i in np.nonzero(unique)[0]:
        try:
            for train_structure in train_by_elements.get(element_set(structures[i]), []):
                if matcher.fit(structures[i], train_structure):
                    novel[i] = False
                    break
        except Exception as e:
            logging.warning(f"Error checking novelty for structure {i}: {e}")
    return unique, novel


def threshold_mask(values, task):
    if task == "space_group":
        return np.array([sg == TASKS[task]["target"] for sg in values])
    low, high = TASKS[task]["threshold"]
    values = np.array(values)
    mask = np.ones(len(values), dtype=bool)
    if low is not None:
        mask &= values >= low
    if high is not None:
        mask &= values <= high
    return mask


def open_pickle(path):
    if str(path).endswith(".gz"):
        import gzip

        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    setup_logging()
    start_time = time.time()
    use_space_group = args.task == "space_group"

    samples = load_samples(args.samples, use_space_group)
    num_samples = samples["atom_types"].shape[0]
    logging.info(f"Loaded {num_samples} samples from {len(args.samples)} file(s)")

    # 1. validity
    structures, values = [], []
    for i in range(num_samples):
        valid, reason = is_valid(
            samples["atom_types"][i].numpy(),
            samples["frac_coords"][i].numpy(),
            samples["lengths"][i].numpy(),
            samples["angles"][i].numpy(),
            samples["lattices"][i].numpy(),
        )
        if not valid:
            logging.debug(f"Sample {i} invalid: {reason}")
            continue
        structures.append(
            to_structure(samples["atom_types"][i], samples["frac_coords"][i], samples["lattices"][i])
        )
        if use_space_group:
            values.append(samples["space_groups"][i])
        else:
            values.append(float(samples["property"][i]))
    num_valid = len(structures)
    logging.info(f"Valid samples: {num_valid}/{num_samples}")

    # 2. stability after CHGNet relaxation
    ppd_mp = open_pickle(args.ppd_path)
    relaxer = StructOptimizer(model=CHGNet.load())
    e_above_hull = []
    for i, structure in enumerate(structures):
        e_above_hull.append(
            energy_above_hull(structure, ppd_mp, relaxer, args.relaxation_steps)
        )
        if (i + 1) % args.log_interval == 0:
            logging.info(f"Relaxed {i + 1}/{num_valid} structures")
    stable = np.array(e_above_hull) < args.stability_threshold

    # 3. uniqueness and novelty
    train_by_elements = load_train_structures(args.train_csv)
    unique, unique_novel = uniqueness_and_novelty(structures, train_by_elements)

    # 4. threshold
    target_met = threshold_mask(values, args.task)

    sun = stable & unique_novel
    sunt = sun & target_met
    results = {
        "task": args.task,
        "samples": args.samples,
        "num_samples": num_samples,
        "valid": num_valid / num_samples,
        "stable": int(stable.sum()) / num_samples,
        "unique": int(unique.sum()) / num_samples,
        "unique_novel": int(unique_novel.sum()) / num_samples,
        "threshold": int(target_met.sum()) / num_samples,
        "sun": int(sun.sum()) / num_samples,
        "sunt": int(sunt.sum()) / num_samples,
        "e_above_hull": [float(e) for e in e_above_hull],
    }

    print("=" * 60)
    print(f"S.U.N.T. results for {args.task} ({num_samples} samples)")
    print("=" * 60)
    for key, name in [
        ("valid", "Valid"),
        ("stable", "S."),
        ("unique", "U."),
        ("unique_novel", "U.N."),
        ("threshold", "T."),
        ("sun", "S.U.N."),
        ("sunt", "S.U.N.T."),
    ]:
        print(f"{name:10s} {100 * results[key]:5.1f}%")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} min")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
