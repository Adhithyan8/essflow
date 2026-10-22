"""ESS-Flow for designable protein backbones with a Proteina prior.

Samples the reward-tilted prior with log g(x) = -w * scRMSD(x), where the
scRMSD is computed with one ProteinMPNN sequence during sampling. The final
samples are evaluated with eight sequences.

Example:
    python -m experiments.proteins.essflow --weight 1.0 \
        --output results/proteins/essflow_w1.0.pt
"""

import argparse
import logging
import time

import torch

from essflow import EllipticalSliceSampler
from experiments.common import get_device, save, seed_everything, setup_logging
from experiments.proteins.proteina import DEFAULT_CKPT, generate, load_model
from experiments.proteins.scrmsd import ScRMSDOracle


def parse_args():
    parser = argparse.ArgumentParser(description="ESS-Flow protein backbone designability")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--n-residues", type=int, default=100)
    parser.add_argument("--weight", type=float, default=1.0,
                        help="reward strength w in log g(x) = -w * scRMSD(x)")
    parser.add_argument("--num-seq", type=int, default=1,
                        help="ProteinMPNN sequences per structure during sampling")
    parser.add_argument("--eval-num-seq", type=int, default=8,
                        help="ProteinMPNN sequences per structure for the final evaluation")
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--num-mcmc-samples", type=int, default=64,
                        help="samples kept per chain after burn-in")
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--max-subiter", type=int, default=20)
    parser.add_argument("--logging-interval", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=16)
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
    model = load_model(args.ckpt, device)
    oracle = ScRMSDOracle(num_seq=args.num_seq, temp=0.1, device=device)

    def log_likelihood_fn(z):
        atom37 = generate(model, z.shape[0], args.n_residues, source=z)
        scrmsd = oracle(atom37)
        logging.debug(f"scRMSD {', '.join(f'{v:.2f}' for v in scrmsd)}")
        return -args.weight * torch.tensor(scrmsd, dtype=torch.float32, device=z.device)

    logging.info(
        f"Running {args.num_chains} chains x {args.num_mcmc_samples} samples "
        f"(+{args.num_warmup} warmup), w={args.weight}, {args.n_residues} residues"
    )
    init_z = torch.randn(args.num_chains, args.n_residues, 3, device=device)
    sampler = EllipticalSliceSampler(
        log_likelihood_fn,
        max_subiter=args.max_subiter,
        logging_interval=args.logging_interval,
    )
    samples, nfe = sampler.sample(
        init_z, num_samples=args.num_mcmc_samples, num_warmup=args.num_warmup
    )
    logging.info(f"Sampling done in {(time.time() - start_time) / 60:.1f} min")

    # decode and evaluate all retained samples, ordered chain by chain
    oracle.num_seq = args.eval_num_seq
    source = samples.reshape(-1, args.n_residues, 3)
    atom37, scrmsd = [], []
    for start in range(0, source.shape[0], args.eval_batch_size):
        z = source[start : start + args.eval_batch_size]
        a37 = generate(model, z.shape[0], args.n_residues, source=z)
        atom37.append(a37.cpu())
        scrmsd.extend(oracle(a37))
    scrmsd = torch.tensor(scrmsd)

    logging.info(
        f"scRMSD mean {scrmsd.mean():.2f} | designable (< 2 A) "
        f"{100 * (scrmsd < 2.0).float().mean():.1f}% | NFE per chain mean {nfe.float().mean():.1f}"
    )
    save(
        {
            "atom37": torch.cat(atom37),
            "scrmsd": scrmsd,
            "nfe": nfe.cpu(),
            "source": source.cpu(),
            "config": vars(args),
        },
        args.output,
    )
    logging.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
