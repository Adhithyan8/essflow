"""Unconditional samples from FlowMM, evaluated on a material generation task."""

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
    TASKS,
    get_space_groups,
    load_property_model,
    predict_property,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional FlowMM samples")
    add_prior_args(parser)
    parser.add_argument("--task", type=str, choices=list(TASKS), required=True)
    parser.add_argument("--num-atoms", type=int, default=None,
                        help="atoms per unit cell (default: the value used in the paper)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tol", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.num_atoms is None:
        args.num_atoms = TASKS[args.task]["num_atoms"]
    return args


@torch.no_grad()
def main():
    args = parse_args()
    setup_logging(not args.quiet)
    seed_everything(args.seed)
    start_time = time.time()

    device = get_device()
    memory_tracker = GPUMemoryTracker(device)
    prior = FlowMMPrior(args, args.num_atoms, device)
    property_model = None
    if args.task != "space_group":
        property_model = load_property_model(args.task)

    outputs = []
    for start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - start)
        logging.info(f"Generating samples {start + 1}-{start + batch_size}")
        z = prior.sample_source(batch_size)
        recon = prior.generate(z, num_steps=args.ode_steps)
        crystals = prior.to_crystals(recon)
        out = crystals_to_dict(*crystals)
        if args.task == "space_group":
            space_groups = get_space_groups(
                *crystals, symprec=args.symprec, angle_tolerance=args.angle_tol
            )
            out["space_groups"] = [sg if sg is not None else "Unknown" for sg in space_groups]
        else:
            out["property"] = predict_property(*crystals, property_model, device).cpu()
        outputs.append(out)
    memory_tracker.record("After Generation")

    combined = {
        k: (sum((o[k] for o in outputs), []) if k == "space_groups"
            else torch.cat([o[k] for o in outputs]))
        for k in outputs[0]
    }
    combined["config"] = vars(args)
    if "property" in combined:
        logging.info(
            f"Property stats - Mean: {combined['property'].mean():.4f}, "
            f"Std: {combined['property'].std():.4f}"
        )
    save(combined, args.output)
    logging.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    memory_tracker.print_summary()


if __name__ == "__main__":
    main()
