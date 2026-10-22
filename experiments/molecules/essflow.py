"""ESS-Flow for drug-like molecule generation with a GeoLDM prior.

Samples the reward-tilted prior with log g(x) = w * (QED + SA_norm) / 2 for
connected molecules and -w for disconnected ones.

Example:
    python -m experiments.molecules.essflow --model-path pretrained/geoldm_drugs \
        --reward-weight 10 --output results/molecules/essflow_w10.pt
"""

import argparse
import logging
import time

import torch

from essflow import EllipticalSliceSampler
from experiments.common import (
    GPUMemoryTracker,
    get_device,
    save,
    seed_everything,
    setup_logging,
)
from experiments.molecules.geoldm import (
    decode,
    load_model,
    log_reward,
    sample_source,
    score_molecules,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ESS-Flow molecule generation (QED/SA)")
    parser.add_argument("--model-path", type=str, default="pretrained/geoldm_drugs",
                        help="GeoLDM checkpoint directory (args.pickle, generative_model_ema.npy)")
    parser.add_argument("--reward-weight", type=float, default=10.0)
    parser.add_argument("--num-atoms", type=int, default=28, help="including hydrogens")
    parser.add_argument("--ddim-steps", type=int, default=50)

    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--num-mcmc-samples", type=int, default=64,
                        help="samples kept per chain after burn-in")
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--max-subiter", type=int, default=20)
    parser.add_argument("--logging-interval", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="batch size for decoding the final samples")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(not args.quiet)
    seed_everything(args.seed)
    start_time = time.time()

    device = get_device()
    logging.info(f"Using device: {device}")
    memory_tracker = GPUMemoryTracker(device)

    model, dataset_info = load_model(args.model_path, device)
    max_n_nodes = dataset_info["max_n_nodes"]
    assert args.num_atoms <= max_n_nodes, f"--num-atoms exceeds the model max {max_n_nodes}"
    memory_tracker.record("After Model Load")

    def evaluate(z):
        x, one_hot = decode(z, model, args.num_atoms, max_n_nodes, args.ddim_steps)
        scores = score_molecules(x, one_hot, args.num_atoms, dataset_info)
        return x, one_hot, scores

    call_counter = [0]

    def log_likelihood_fn(z):
        _, _, scores = evaluate(z)
        log_like = torch.tensor(
            [log_reward(qed, sa, args.reward_weight) for qed, sa in scores],
            device=z.device,
        )
        logging.debug(
            f"call {call_counter[0]:4d} | "
            + ", ".join(
                f"qed={q:.3f} sa={s:.2f}" if q is not None else "disconnected"
                for q, s in scores
            )
        )
        call_counter[0] += 1
        return log_like

    logging.info(
        f"Running {args.num_chains} chains x {args.num_mcmc_samples} samples "
        f"(+{args.num_warmup} warmup), w={args.reward_weight}"
    )
    init_z = sample_source(model, args.num_chains, args.num_atoms, device)
    sampler = EllipticalSliceSampler(
        log_likelihood_fn,
        max_subiter=args.max_subiter,
        logging_interval=args.logging_interval,
    )
    samples, nfe = sampler.sample(
        init_z, num_samples=args.num_mcmc_samples, num_warmup=args.num_warmup
    )
    memory_tracker.record("After ESS Sampling")

    # decode and score all retained samples, ordered chain by chain
    source = samples.reshape(args.num_chains * args.num_mcmc_samples, -1)
    positions, one_hots, qeds, sas = [], [], [], []
    for start in range(0, source.shape[0], args.eval_batch_size):
        x, one_hot, scores = evaluate(source[start : start + args.eval_batch_size])
        positions.append(x.cpu())
        one_hots.append(one_hot.cpu())
        qeds.extend(q for q, _ in scores)
        sas.extend(s for _, s in scores)

    connected = [q is not None for q in qeds]
    logging.info(f"Connected: {sum(connected)}/{len(qeds)}")
    logging.info(
        f"NFE per chain - mean: {nfe.float().mean():.1f}, min: {nfe.min()}, max: {nfe.max()}"
    )

    save(
        {
            "positions": torch.cat(positions),
            "one_hot": torch.cat(one_hots),
            "qed": qeds,
            "sa": sas,
            "nfe": nfe.cpu(),
            "num_atoms": args.num_atoms,
            "source": source.cpu(),
            "config": vars(args),
        },
        args.output,
    )
    logging.info(f"Total time: {(time.time() - start_time) / 60:.2f} min")
    memory_tracker.print_summary()


if __name__ == "__main__":
    main()
