"""ESS-Flow for material generation with a FlowMM prior.

Example (bulk modulus, 125 chains, one sample per chain after 200 burn-in steps):
    python -m experiments.materials.essflow --checkpoint $FLOWMM_CKPT \
        --task bulk_modulus --num-chains 125 --output results/materials/bm_essflow_1.pt
"""

import argparse
import logging
import time
from collections import Counter

import torch

from essflow import EllipticalSliceSampler
from experiments.common import (
    GPUMemoryTracker,
    get_device,
    save,
    seed_everything,
    setup_logging,
)
from experiments.materials.prior import FlowMMPrior, add_prior_args, crystals_to_dict
from experiments.materials.tasks import (
    TASKS,
    get_space_groups,
    load_property_model,
    log_potential,
    predict_property,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ESS-Flow material generation")
    add_prior_args(parser)
    parser.add_argument("--task", type=str, choices=list(TASKS), required=True)
    parser.add_argument("--num-atoms", type=int, default=None,
                        help="atoms per unit cell (default: the value used in the paper)")
    parser.add_argument("--num-chains", type=int, default=125)

    # ESS
    parser.add_argument("--num-mcmc-samples", type=int, default=1,
                        help="samples kept per chain after burn-in")
    parser.add_argument("--num-warmup", type=int, default=None,
                        help="burn-in steps (default: 500 for band_gap, otherwise 200)")
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--max-subiter", type=int, default=100)
    parser.add_argument("--logging-interval", type=int, default=10)

    # space group task
    parser.add_argument("--target-sg", type=str, default="P6_3/mmc")
    parser.add_argument("--sg-weight", type=float, default=100.0,
                        help="log-potential penalty for a mismatched space group")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tol", type=float, default=5.0)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.num_atoms is None:
        args.num_atoms = TASKS[args.task]["num_atoms"]
    if args.num_warmup is None:
        args.num_warmup = 500 if args.task == "band_gap" else 200
    return args


def main():
    args = parse_args()
    setup_logging(not args.quiet)
    seed_everything(args.seed)
    start_time = time.time()

    device = get_device()
    logging.info(f"Using device: {device}")
    memory_tracker = GPUMemoryTracker(device)

    prior = FlowMMPrior(args, args.num_atoms, device)
    memory_tracker.record("After Model Load")

    property_model = None
    if args.task != "space_group":
        property_model = load_property_model(args.task)
        memory_tracker.record("After Property Model Load")

    def evaluate(z):
        """Decode source samples and evaluate the task on the crystals."""
        recon = prior.generate(z, num_steps=args.ode_steps)
        crystals = prior.to_crystals(recon)
        if args.task == "space_group":
            space_groups = get_space_groups(
                *crystals, symprec=args.symprec, angle_tolerance=args.angle_tol
            )
            log_like = torch.tensor(
                [0.0 if sg == args.target_sg else -args.sg_weight for sg in space_groups]
            )
            return log_like, crystals, space_groups
        output = predict_property(*crystals, property_model, device).cpu()
        return log_potential(output, args.task), crystals, output

    def log_likelihood_fn(z):
        with torch.no_grad():
            return evaluate(z)[0]

    logging.info(
        f"Task: {args.task} | {args.num_chains} chains x {args.num_mcmc_samples} samples "
        f"(+{args.num_warmup} warmup) | {args.ode_steps} ODE steps | {args.num_atoms} atoms"
    )

    init_z = prior.sample_source(args.num_chains)
    memory_tracker.record("Before ESS Sampling")
    sampler = EllipticalSliceSampler(
        log_likelihood_fn,
        max_subiter=args.max_subiter,
        logging_interval=args.logging_interval,
    )
    samples, nfe = sampler.sample(
        init_z, num_samples=args.num_mcmc_samples, num_warmup=args.num_warmup
    )
    memory_tracker.record("After ESS Sampling")

    source = samples.reshape(-1, prior.dim)
    with torch.no_grad():
        _, crystals, values = evaluate(source)
    memory_tracker.record("After Property Evaluation")

    out = crystals_to_dict(*crystals)
    out["nfe"] = nfe.cpu()
    out["source"] = source.cpu()
    out["config"] = vars(args)
    if args.task == "space_group":
        out["space_groups"] = [sg if sg is not None else "Unknown" for sg in values]
        counts = Counter(out["space_groups"])
        logging.info(f"Top space groups: {counts.most_common(5)}")
        logging.info(
            f"Space group match rate: "
            f"{100 * counts[args.target_sg] / len(out['space_groups']):.1f}%"
        )
    else:
        out["property"] = values
        logging.info(
            f"Property stats - Mean: {values.mean():.4f}, Std: {values.std():.4f}, "
            f"Min: {values.min():.4f}, Max: {values.max():.4f}"
        )
    logging.info(
        f"NFE per chain - Mean: {nfe.float().mean():.2f}, "
        f"Min: {nfe.min().item()}, Max: {nfe.max().item()}"
    )

    save(out, args.output)
    logging.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    memory_tracker.print_summary()


if __name__ == "__main__":
    main()
