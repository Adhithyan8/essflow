"""PnP-Flow baseline: gradient steps in data space with FlowMM noising-denoising.

Each iteration takes a gradient step on the cost in data space (soft atom
embedding), interpolates towards a fresh source sample at time t, and
integrates the ODE from t back to 1. The lowest-cost iterate is returned.
"""

import argparse
import logging
import time

import torch

from experiments.common import (
    GPUMemoryTracker,
    get_device,
    save,
    seed_everything,
    setup_logging,
)
from experiments.materials.dflow import cost_fn
from experiments.materials.prior import FlowMMPrior, add_prior_args, crystals_to_dict
from experiments.materials.tasks import (
    PROPERTY_TASKS,
    TASKS,
    load_property_model,
    predict_property,
)

# max. learning rates for (atom types, frac. coords, lattice lengths, lattice angles),
# decayed linearly to 0
LEARNING_RATES = {
    "bulk_modulus": (1e-4, 1e-6, 1e-5, 1e-5),
    "shear_modulus": (1e-3, 1e-5, 1e-4, 1e-4),
    "band_gap": (1e-1, 1e-3, 1e-2, 1e-2),
    "e_hull": (1e-2, 1e-4, 1e-3, 1e-3),
}


def parse_args():
    parser = argparse.ArgumentParser(description="PnP-Flow material generation")
    add_prior_args(parser)
    parser.add_argument("--task", type=str, choices=PROPERTY_TASKS, required=True)
    parser.add_argument("--num-atoms", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: 125 for band_gap, otherwise 250. The loss is "
                             "averaged over the batch, so the effective step size "
                             "depends on it")
    parser.add_argument("--pnpflow-steps", type=int, default=100)
    parser.add_argument("--ode-init-steps", type=int, default=50)
    parser.add_argument("--lr-atom", type=float, default=None)
    parser.add_argument("--lr-frac", type=float, default=None)
    parser.add_argument("--lr-length", type=float, default=None)
    parser.add_argument("--lr-angle", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.num_atoms is None:
        args.num_atoms = TASKS[args.task]["num_atoms"]
    if args.batch_size is None:
        args.batch_size = 125 if args.task == "band_gap" else 250
    defaults = LEARNING_RATES[args.task]
    for name, lr in zip(["lr_atom", "lr_frac", "lr_length", "lr_angle"], defaults):
        if getattr(args, name) is None:
            setattr(args, name, lr)
    return args


def pnpflow(prior, property_model, batch_size, args, device, memory_tracker):
    n = args.num_atoms
    sizes = [7 * n, 3 * n, 3, 3]
    lrs = [args.lr_atom, args.lr_frac, args.lr_length, args.lr_angle]

    recon = prior.generate(prior.sample_source(batch_size), args.ode_init_steps)
    recon = recon.detach().requires_grad_(True)
    manifold = prior.manifold(batch_size)

    best_cost = torch.full((batch_size,), float("inf"), device=recon.device)
    best_recon = torch.zeros_like(recon)

    for step in range(args.pnpflow_steps):
        cost = cost_fn(
            predict_property(
                *prior.to_crystals(recon), property_model, device, soft_atom_types=True
            ),
            args.task,
        )
        with torch.no_grad():
            improved = cost < best_cost
            best_cost = torch.where(improved, cost, best_cost)
            best_recon = torch.where(improved.unsqueeze(1), recon, best_recon)
        cost.mean().backward()

        with torch.no_grad():
            t = torch.tensor(step / args.pnpflow_steps, device=recon.device)
            t_batch = torch.full((batch_size, 1), t, device=recon.device)

            # gradient step with linearly decayed, per-component learning rates
            recon_hat = torch.cat(
                [
                    x - lr * (1 - t) * g
                    for x, g, lr in zip(
                        torch.split(recon, sizes, dim=1),
                        torch.split(recon.grad, sizes, dim=1),
                        lrs,
                    )
                ],
                dim=1,
            )

            # interpolate towards a fresh source sample, then denoise from t to 1
            x0 = prior.to_base(prior.sample_source(batch_size)).to(recon.device)
            x_t, _ = manifold.cond_u(x0, recon_hat, t_batch)
            recon = prior.generate_from_xt(
                t, x_t, num_steps=args.pnpflow_steps - step
            )
        recon = recon.detach().requires_grad_(True)

        if step < 2:
            memory_tracker.record(f"PnP-Flow Step {step}")

    return best_recon.detach()


def main():
    args = parse_args()
    setup_logging(not args.quiet)
    seed_everything(args.seed)
    start_time = time.time()

    device = get_device()
    memory_tracker = GPUMemoryTracker(device)
    prior = FlowMMPrior(args, args.num_atoms, device)
    property_model = load_property_model(args.task)
    logging.info(
        f"PnP-Flow: {args.pnpflow_steps} steps, {args.ode_init_steps} initial ODE steps, "
        f"learning rates: atom={args.lr_atom}, frac={args.lr_frac}, "
        f"length={args.lr_length}, angle={args.lr_angle}"
    )

    outputs = []
    for start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - start)
        recon = pnpflow(prior, property_model, batch_size, args, device, memory_tracker)
        with torch.no_grad():
            crystals = prior.to_crystals(recon)
            out = crystals_to_dict(*crystals)
            out["property"] = predict_property(*crystals, property_model, device).cpu()
        outputs.append(out)
        logging.info(
            f"Generated {start + batch_size}/{args.num_samples} samples, "
            f"batch property mean: {out['property'].mean():.4f}"
        )

    combined = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    combined["config"] = vars(args)
    logging.info(
        f"Property mean: {combined['property'].mean():.4f}, "
        f"std: {combined['property'].std():.4f}"
    )
    save(combined, args.output)
    logging.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    memory_tracker.print_summary()


if __name__ == "__main__":
    main()
