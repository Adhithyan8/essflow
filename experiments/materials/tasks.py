"""Material generation tasks: property predictors, potentials and space groups."""

import io
import logging
from pathlib import Path

import dgl
import torch
from torch_geometric.data import Data

from experiments.common import REPO_DIR
from experiments.materials.graph import (
    PBCGraph,
    analog_binary_to_decimal,
    get_pbc_distances,
    radius_graph_pbc,
)

# Material generation tasks in the paper. ``threshold`` is the (min, max) range
# of the predicted property used for the T. criterion of the S.U.N.T. rate.
TASKS = {
    "bulk_modulus": dict(
        short="bm", alignn="jv_bulk_modulus_kv_alignn", num_atoms=4,
        target=300.0, scale=10.0, threshold=(285.0, 315.0),
    ),
    "shear_modulus": dict(
        short="sm", alignn="jv_shear_modulus_gv_alignn", num_atoms=4,
        target=200.0, scale=10.0, threshold=(190.0, 210.0),
    ),
    "band_gap": dict(
        short="bg", alignn="jv_mbj_bandgap_alignn", num_atoms=12,
        target=10.0, scale=0.1, threshold=(9.5, 10.5),
    ),
    "e_hull": dict(
        short="eh", alignn="jv_ehull_alignn", num_atoms=8,
        target=None, scale=0.01, threshold=(None, 0.0),
    ),
    "space_group": dict(
        short="sg", alignn=None, num_atoms=8,
        target="P6_3/mmc", scale=None, threshold=None,
    ),
}
PROPERTY_TASKS = [t for t in TASKS if TASKS[t]["alignn"] is not None]


def log_potential(output, task):
    """log g(c): Gaussian around the target, or exp(-E_hull / scale)."""
    cfg = TASKS[task]
    if task == "e_hull":
        return -output / cfg["scale"]
    return -0.5 * (output - cfg["target"]) ** 2 / cfg["scale"] ** 2


def load_property_model(task, model_dir=REPO_DIR / "pretrained" / "alignn"):
    """Pretrained JARVIS-DFT ALIGNN model of a task.

    Loads ``<model_dir>/<model name>.zip`` in the same way as
    ``alignn.pretrained.get_figshare_model``, which instead downloads the zip
    from figshare (this download can be blocked for non-browser clients).
    """
    import json
    import zipfile

    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    from alignn.pretrained import get_figshare_model

    name = TASKS[task]["alignn"]
    path = Path(model_dir) / f"{name}.zip"
    if not path.is_file():
        logging.info(f"{path} not found, downloading {name} from figshare")
        try:
            return get_figshare_model(name)
        except zipfile.BadZipFile:
            raise RuntimeError(
                f"Downloading {name} from figshare failed. Download it manually "
                f"(see the README) and place it at {path}"
            )

    with zipfile.ZipFile(path) as zp:
        names = zp.namelist()
        # same checkpoint selection as get_figshare_model
        for n in names:
            if ("checkpoint_" in n and "pt" in n) or "best_model.pt" in n:
                checkpoint = n
            if "config.json" in n:
                config = n
        logging.info(f"Loading {checkpoint} from {path}")
        model = ALIGNN(ALIGNNConfig(**json.loads(zp.read(config))["model"]))
        with zp.open(checkpoint) as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def predict_property(
    atom_bits, frac_coords, lattices, property_model, property_device,
    soft_atom_types=False,
):
    """Predict a property with ALIGNN for a batch of crystals.

    Args:
        atom_bits: analog bits of the atom types ``[B, n, 7]``.
        frac_coords: fractional coordinates ``[B, n, 3]``.
        lattices: lattice matrices ``[B, 3, 3]``.
        soft_atom_types: use the differentiable soft atom embedding instead of rounding to discrete atomic numbers.

    Returns:
        Predicted property ``[B]`` on ``property_device``.
    """
    batch_size, num_atoms, _ = atom_bits.shape
    device = atom_bits.device

    cart_coords = torch.matmul(frac_coords, lattices).view(-1, 3)
    data = Data(
        pos=cart_coords,
        cell=lattices,
        natoms=torch.tensor([num_atoms] * batch_size, device=device),
        pbc=torch.tensor([[True, True, True]] * batch_size, device=device),
    )
    edge_index, cell_offsets, neighbors = radius_graph_pbc(
        data,
        radius=8.0,
        max_num_neighbors_threshold=12,
        enforce_max_neighbors_strictly=True,
        pbc=[True, True, True],
    )
    result = get_pbc_distances(
        data.pos, edge_index, data.cell, cell_offsets, neighbors,
        return_distance_vec=True,
    )

    graphs, line_graphs = [], []
    for i in range(batch_size):
        mask = (edge_index[0] >= i * num_atoms) & (edge_index[0] < (i + 1) * num_atoms)
        g, lg = PBCGraph.atom_dgl_multigraph_pbc(
            atom_bits[i],
            frac_coords[i],
            edge_index[:, mask] - i * num_atoms,
            result["distance_vec"][mask],
            cell_offsets[mask],
            soft_atom_types=soft_atom_types,
        )
        graphs.append(g)
        line_graphs.append(lg)

    output = property_model(
        [
            dgl.batch(graphs).to(property_device),
            dgl.batch(line_graphs).to(property_device),
            lattices.to(property_device),
        ]
    )
    return output.reshape(-1)


def get_space_group(atom_bits, frac_coords, lattice, symprec=0.1, angle_tolerance=5.0):
    """Space group symbol and number of one crystal with spglib (via pymatgen)."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    atomic_numbers = analog_binary_to_decimal(atom_bits).cpu().numpy().astype(int)
    try:
        structure = Structure(
            lattice=lattice.detach().cpu().numpy(),
            species=atomic_numbers.flatten(),
            coords=frac_coords.detach().cpu().numpy(),
            coords_are_cartesian=False,
        )
        sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        return sga.get_space_group_symbol(), sga.get_space_group_number()
    except Exception as e:
        logging.debug(f"Space group determination failed: {e}")
        return None, None


def get_space_groups(atom_bits, frac_coords, lattices, symprec=0.1, angle_tolerance=5.0):
    symbols = []
    for i in range(atom_bits.shape[0]):
        symbol, _ = get_space_group(
            atom_bits[i], frac_coords[i], lattices[i], symprec, angle_tolerance
        )
        symbols.append(symbol)
    return symbols
