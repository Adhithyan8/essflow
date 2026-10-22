"""Multi-fidelity ESS-Flow: importance reweighting of coarse ESS-Flow samples.

ESS-Flow samples obtained with a coarse ODE discretization (``--ode-steps-low``)
are reweighted towards the target defined with a fine discretization
(``--ode-steps-high``) using self-normalized importance weights
w ∝ g(T_fine(z)) / g(T_coarse(z)), and resampled with replacement.
"""

import argparse
import logging
import time

import torch

from experiments.common import get_device, save, seed_everything, setup_logging
from experiments.materials.prior import FlowMMPrior, add_prior_args, crystals_to_dict
from experiments.materials.tasks import (
    PROPERTY_TASKS,
    TASKS,
    load_property_model,
    log_potential,
    predict_property,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-fidelity importance reweighting")
    add_prior_args(parser)
    parser.add_argument("--task", type=str, choices=PROPERTY_TASKS, required=True)
    parser.add_argument("--num-atoms", type=int, default=None)
    parser.add_argument("--samples", type=str, nargs="+", required=True,
                        help="ESS-Flow output files (with the source samples)")
    parser.add_argument("--ode-steps-low", type=int, default=50)
    parser.add_argument("--ode-steps-high", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-resamples", type=int, default=None,
                        help="default: the number of ESS-Flow samples")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True,
                        help="resampled high-fidelity samples")
    parser.add_argument("--weights-output", type=str, default=None,
                        help="importance weights (default: <output>_weights.pt)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.num_atoms is None:
        args.num_atoms = TASKS[args.task]["num_atoms"]
    if args.weights_output is None:
        args.weights_output = args.output.replace(".pt", "_weights.pt")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    setup_logging(not args.quiet)
    seed_everything(args.seed)
    start_time = time.time()

    device = get_device()
    prior = FlowMMPrior(args, args.num_atoms, device)
    property_model = load_property_model(args.task)

    source = torch.cat([torch.load(f)["source"] for f in args.samples])
    num_samples = source.shape[0]
    logging.info(
        f"Reweighting {num_samples} samples from {len(args.samples)} file(s): "
        f"ODE steps low={args.ode_steps_low}, high={args.ode_steps_high}"
    )

    log_like_low, log_like_high, high_fidelity = [], [], []
    for start in range(0, num_samples, args.batch_size):
        z = source[start : start + args.batch_size]
        for steps, log_likes in [
            (args.ode_steps_low, log_like_low),
            (args.ode_steps_high, log_like_high),
        ]:
            crystals = prior.to_crystals(prior.generate(z, num_steps=steps))
            output = predict_property(*crystals, property_model, device).cpu()
            log_likes.append(log_potential(output, args.task))
        out = crystals_to_dict(*crystals)
        out["property"] = output
        high_fidelity.append(out)
        logging.info(f"Processed {min(start + args.batch_size, num_samples)}/{num_samples}")

    log_like_low = torch.cat(log_like_low)
    log_like_high = torch.cat(log_like_high)
    high_fidelity = {k: torch.cat([o[k] for o in high_fidelity]) for k in high_fidelity[0]}

    # self-normalized importance weights and effective sample size
    log_w = log_like_high - log_like_low
    weights = torch.softmax(log_w.double(), dim=0)
    ess = 1.0 / torch.sum(weights**2).item()
    logging.info(
        f"Effective sample size: {ess:.2f} ({100 * ess / num_samples:.1f}% of {num_samples})"
    )

    num_resamples = args.num_resamples or num_samples
    indices = torch.multinomial(weights, num_samples=num_resamples, replacement=True)
    resampled = {k: v[indices] for k, v in high_fidelity.items()}
    resampled["config"] = vars(args)
    logging.info(
        f"Unique samples after resampling: {len(torch.unique(indices))}/{num_resamples}"
    )

    save(
        {
            "log_like_low": log_like_low,
            "log_like_high": log_like_high,
            "weights": weights.float(),
            "ess": ess,
            "normalized_ess": ess / num_samples,
            "resampled_indices": indices,
            "config": vars(args),
        },
        args.weights_output,
    )
    save(resampled, args.output)
    logging.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
