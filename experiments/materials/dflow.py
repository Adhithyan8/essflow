"""D-Flow baseline: gradient-based optimization of the FlowMM source.

The cost is differentiated through the ODE solver and through ALIGNN, using the
soft atom embedding for the discrete atom types. The source sample with the
lowest cost seen during optimization is returned.
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
from experiments.materials.prior import FlowMMPrior, add_prior_args, crystals_to_dict
from experiments.materials.tasks import (
    PROPERTY_TASKS,
    TASKS,
    load_property_model,
    predict_property,
)

# learning rates for (atom types, frac. coords, lattice lengths, lattice angles)
LEARNING_RATES = {
    "bulk_modulus": (1e-6, 1e-8, 1e-6, 1e-6),
    "shear_modulus": (1e-5, 1e-7, 1e-5, 1e-5),
    "band_gap": (1e-4, 1e-6, 1e-4, 1e-4),
    "e_hull": (1e-4, 1e-6, 1e-4, 1e-4),
}


def parse_args():
    parser = argparse.ArgumentParser(description="D-Flow material generation")
    add_prior_args(parser)
    parser.add_argument("--task", type=str, choices=PROPERTY_TASKS, required=True)
    parser.add_argument("--num-atoms", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50,
                        help="the loss is averaged over the batch, so the effective "
                             "step size depends on it")
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--optim-steps", type=int, default=100)
    parser.add_argument("--lr-atom", type=float, default=None)
    parser.add_argument("--lr-frac", type=float, default=None)
    parser.add_argument("--lr-length", type=float, default=None)
    parser.add_argument("--lr-angle", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="retries of a batch whose ODE solution became NaN")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.num_atoms is None:
        args.num_atoms = TASKS[args.task]["num_atoms"]
    defaults = LEARNING_RATES[args.task]
    for name, lr in zip(["lr_atom", "lr_frac", "lr_length", "lr_angle"], defaults):
        if getattr(args, name) is None:
            setattr(args, name, lr)
    return args


def cost_fn(output, task):
    """Unnormalized cost used by D-Flow and PnP-Flow (learning rates are tuned for it)."""
    if task == "e_hull":
        return 100.0 * output
    return (output - TASKS[task]["target"]) ** 2


def dflow(prior, property_model, batch_size, args, device, memory_tracker):
    n = args.num_atoms
    sizes = [7 * n, 3 * n, 3, 3]
    params = [
        p.detach().requires_grad_(True)
        for p in torch.split(prior.sample_source(batch_size), sizes, dim=1)
    ]
    lrs = [args.lr_atom, args.lr_frac, args.lr_length, args.lr_angle]
    optimizer = torch.optim.SGD([{"params": [p], "lr": lr} for p, lr in zip(params, lrs)])

    best_costs = torch.full((batch_size,), float("inf"), device=device)
    best_params = [torch.zeros_like(p) for p in params]

    start_time = time.time()
    for step in range(args.optim_steps):
        optimizer.zero_grad()
        recon = prior.generate(torch.cat(params, dim=1), args.ode_steps, with_grad=True)
        if torch.isnan(recon).any():
            raise FloatingPointError("NaN values during optimization")
        costs = cost_fn(
            predict_property(
                *prior.to_crystals(recon), property_model, device, soft_atom_types=True
            ),
            args.task,
        )
        loss = costs.mean()
        loss.backward()

        # keep the source point with the lowest cost for each sample
        with torch.no_grad():
            improved = (costs < best_costs).cpu()
            best_costs[improved] = costs[improved]
            for best, p in zip(best_params, params):
                best[improved] = p[improved]

        optimizer.step()
        if step < 2:
            memory_tracker.record(f"Optimization Step {step}")
        logging.info(
            f"Step {step + 1}/{args.optim_steps}, Loss: {loss.item():.4f}, "
            f"Time: {time.time() - start_time:.2f} sec"
        )

    with torch.no_grad():
        return prior.generate(torch.cat(best_params, dim=1), args.ode_steps)


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
        f"Learning rates: atom={args.lr_atom}, frac={args.lr_frac}, "
        f"length={args.lr_length}, angle={args.lr_angle}"
    )

    outputs = []
    failed_batches = 0
    for start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - start)
        for attempt in range(args.max_retries + 1):
            try:
                recon = dflow(prior, property_model, batch_size, args, device, memory_tracker)
                break
            except FloatingPointError:
                logging.warning(f"NaN values during optimization (attempt {attempt + 1})")
        else:
            failed_batches += 1
            logging.error(f"Batch at sample {start} failed, skipping it")
            continue

        with torch.no_grad():
            crystals = prior.to_crystals(recon)
            out = crystals_to_dict(*crystals)
            out["property"] = predict_property(*crystals, property_model, device).cpu()
        outputs.append(out)
        logging.info(f"Generated {start + batch_size}/{args.num_samples} samples")

    combined = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    combined["config"] = vars(args)
    logging.info(
        f"Generated {len(combined['property'])} samples ({failed_batches} failed batches). "
        f"Property mean: {combined['property'].mean():.4f}, "
        f"std: {combined['property'].std():.4f}"
    )
    save(combined, args.output)
    logging.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    memory_tracker.print_summary()


if __name__ == "__main__":
    main()
