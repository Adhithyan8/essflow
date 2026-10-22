"""Unconditional Proteina samples (probability flow ODE), evaluated for designability."""

import argparse
import logging
import time

import torch

from experiments.common import get_device, save, seed_everything, setup_logging
from experiments.proteins.proteina import DEFAULT_CKPT, generate, load_model
from experiments.proteins.scrmsd import ScRMSDOracle


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional Proteina samples")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--n-residues", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-seq", type=int, default=8,
                        help="ProteinMPNN sequences per structure")
    parser.add_argument("--batch-size", type=int, default=16)
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
    model = load_model(args.ckpt, device)
    oracle = ScRMSDOracle(num_seq=args.num_seq, temp=0.1, device=device)

    atom37, scrmsd = [], []
    for start in range(0, args.num_samples, args.batch_size):
        bs = min(args.batch_size, args.num_samples - start)
        a37 = generate(model, bs, args.n_residues)
        atom37.append(a37.cpu())
        scrmsd.extend(oracle(a37))
        logging.info(f"{start + bs}/{args.num_samples} | scRMSD {scrmsd[-bs:]}")
    scrmsd = torch.tensor(scrmsd)

    logging.info(
        f"scRMSD mean {scrmsd.mean():.2f} | designable (< 2 A) "
        f"{100 * (scrmsd < 2.0).float().mean():.1f}%"
    )
    save({"atom37": torch.cat(atom37), "scrmsd": scrmsd, "config": vars(args)}, args.output)
    logging.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
