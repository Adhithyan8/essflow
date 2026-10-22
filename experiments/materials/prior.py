"""FlowMM as a transport map from a standard Gaussian source to crystals."""

import numpy as np
import torch
from diffcsp.common.data_utils import lattices_to_params_shape
from diffcsp.script_utils import GenDataset
from torch_geometric.data import DataLoader

from flowmm.model.eval_utils import load_model, register_omega_conf_resolvers
from flowmm.rfm.manifolds.lattice_params import get_lattice_params_stats

from experiments.materials.graph import analog_binary_to_decimal


class CondDataset(GenDataset):
    """FlowMM's empty generation dataset with a fixed number of atoms."""

    def __init__(self, dataset, total_num, specified_num_atoms):
        super().__init__(dataset, total_num)
        self.num_atoms = np.array([specified_num_atoms] * total_num)


def constrained_to_unconstrained(x, loc, scale):
    # affine inv
    x = (x - loc) / scale
    # sigmoid inv
    return torch.log(x / (1 - x))


def gaussian_to_lognormal(x, loc, scale):
    return torch.exp(x * scale + loc)


def gaussian_to_uniform_range(x, low, high):
    u01 = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    return u01 * (high - low) + low


def add_prior_args(parser):
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mp_20")
    # lattice angle transforms of the FlowMM base distribution
    parser.add_argument("--angle-low", type=float, default=59.9)
    parser.add_argument("--angle-high", type=float, default=120.1)
    parser.add_argument("--transform-angle-loc", type=float, default=59.9)
    parser.add_argument("--transform-angle-scale", type=float, default=60.2)


class FlowMMPrior:
    """Pretrained FlowMM with a standard Gaussian source.

    A crystal with ``n`` atoms is represented in the source space by a flat
    vector of size ``7n + 3n + 3 + 3`` (analog bits, fractional coordinates,
    lattice lengths, lattice angles). The uniform and log-normal parts of the
    FlowMM base distribution are obtained from the Gaussian by a change of
    variables, so that the full source distribution is N(0, I).
    """

    def __init__(self, args, num_atoms, device):
        self.num_atoms = num_atoms
        self.device = device
        self.angle_low = args.angle_low
        self.angle_high = args.angle_high
        self.transform_angle_loc = args.transform_angle_loc
        self.transform_angle_scale = args.transform_angle_scale

        register_omega_conf_resolvers()
        stats = get_lattice_params_stats(args.dataset)
        self.length_loc = stats.length_log_mean.unsqueeze(0)
        self.length_scale = stats.length_log_std.unsqueeze(0)

        self.cfg, self.model = load_model(args.checkpoint)
        self.model.to(device)
        self._batches = {}

    @property
    def dim(self):
        return self.num_atoms * 10 + 6

    def empty_batch(self, batch_size):
        """FlowMM batch structure for ``batch_size`` crystals (cached)."""
        if batch_size not in self._batches:
            sample_set = CondDataset(
                dataset=self.cfg.data.dataset_name,
                total_num=batch_size,
                specified_num_atoms=self.num_atoms,
            )
            batch = next(iter(DataLoader(sample_set, batch_size=batch_size)))
            batch.to(self.device)
            *_, dims, mask_a_or_f = self.model.manifold_getter.from_empty_batch(
                batch.batch, dim_coords=3, split_manifold=False
            )
            self._batches[batch_size] = (batch, dims, mask_a_or_f)
        return self._batches[batch_size]

    def manifold(self, batch_size):
        batch, _, _ = self.empty_batch(batch_size)
        return self.model.manifold_getter.from_empty_batch(
            batch.batch, dim_coords=3, split_manifold=True
        )[1]

    def sample_source(self, batch_size):
        n = self.num_atoms
        l_atom_types = torch.randn(batch_size, n * 7)
        l_frac_coords = torch.randn(batch_size, n * 3)
        l_lengths = torch.randn(batch_size, 3)
        l_angles = torch.randn(batch_size, 3)
        return torch.cat([l_atom_types, l_frac_coords, l_lengths, l_angles], dim=1)

    def to_base(self, z):
        """Map Gaussian source samples to the FlowMM base distribution."""
        n = self.num_atoms
        l_atom_types, l_frac_coords, l_lengths, l_angles = torch.split(
            z, [n * 7, n * 3, 3, 3], dim=1
        )
        l_frac_coords = gaussian_to_uniform_range(l_frac_coords, 0, 1)
        l_lengths = gaussian_to_lognormal(
            l_lengths,
            loc=self.length_loc.to(z.device),
            scale=self.length_scale.to(z.device),
        )
        l_angles = gaussian_to_uniform_range(
            l_angles, low=self.angle_low, high=self.angle_high
        )
        l_angles = constrained_to_unconstrained(
            l_angles, self.transform_angle_loc, self.transform_angle_scale
        )
        return torch.cat([l_atom_types, l_frac_coords, l_lengths, l_angles], dim=1)

    def generate(self, z, num_steps, with_grad=False):
        """Transport Gaussian source samples to the flat crystal representation."""
        batch, _, _ = self.empty_batch(z.shape[0])
        x0 = self.to_base(z)
        gen = self.model.gen_sample_with_grad if with_grad else self.model.gen_sample
        return gen(batch.batch, dim_coords=3, num_steps=num_steps, x0=x0)

    def generate_from_xt(self, t, xt, num_steps):
        """Integrate the ODE from time ``t`` to 1, starting at ``xt``."""
        batch, _, _ = self.empty_batch(xt.shape[0])
        return self.model.gen_sample_xt(
            batch.batch, dim_coords=3, t=t, xt=xt, num_steps=num_steps
        )

    def to_crystals(self, recon):
        """Split the flat representation into per-crystal tensors.

        Returns analog atom bits ``[B, n, 7]``, fractional coordinates
        ``[B, n, 3]`` and lattice matrices ``[B, 3, 3]``.
        """
        _, dims, mask_a_or_f = self.empty_batch(recon.shape[0])
        atom_types, frac_coords, lattices = self.model.manifold_getter.flatrep_to_crystal(
            recon, dims, mask_a_or_f
        )
        batch_size = lattices.shape[0]
        n = atom_types.shape[0] // batch_size
        return (
            atom_types.view(batch_size, n, -1),
            frac_coords.view(batch_size, n, 3),
            lattices,
        )


def crystals_to_dict(atom_bits, frac_coords, lattices):
    """Output format shared by all material scripts (consumed by evaluation)."""
    batch_size, n, _ = atom_bits.shape
    lengths, angles = lattices_to_params_shape(lattices)
    atom_types = analog_binary_to_decimal(atom_bits.reshape(batch_size * n, -1))
    return {
        "atom_types": atom_types.view(batch_size, n).cpu(),
        "frac_coords": frac_coords.cpu(),
        "lattices": lattices.cpu(),
        "lengths": lengths.cpu(),
        "angles": angles.cpu(),
    }
