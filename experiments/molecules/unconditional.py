"""Unconditional GeoLDM samples with the DDPM (1000 steps) or DDIM sampler."""

import argparse
import logging
import time

import torch

from experiments.common import get_device, save, seed_everything, setup_logging
from experiments.molecules.geoldm import build_masks, load_model, score_molecules


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional GeoLDM samples")
    parser.add_argument("--model-path", type=str, default="pretrained/geoldm_drugs")
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-atoms", type=int, default=28, help="including hydrogens")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    setup_logging(not args.quiet)
    seed_everything(args.seed)
    start_time = time.time()

    device = get_device()
    model, dataset_info = load_model(args.model_path, device)
    max_n_nodes = dataset_info["max_n_nodes"]

    positions, one_hots, qeds, sas = [], [], [], []
    for start in range(0, args.num_samples, args.batch_size):
        bs = min(args.batch_size, args.num_samples - start)
        node_mask, edge_mask = build_masks(bs, args.num_atoms, max_n_nodes, device)
        if args.sampler == "ddpm":
            x, h = model.sample(bs, max_n_nodes, node_mask, edge_mask, context=None)
        else:
            z_T = model.sample_combined_position_feature_noise(bs, max_n_nodes, node_mask)
            x, h = model.sample_ddim(
                z_T, node_mask, edge_mask, context=None, num_steps=args.ddim_steps
            )
        one_hot = h["categorical"]
        for qed, sa in score_molecules(x, one_hot, args.num_atoms, dataset_info):
            qeds.append(qed)
            sas.append(sa)
        positions.append(x.cpu())
        one_hots.append(one_hot.cpu())
        logging.info(
            f"{start + bs}/{args.num_samples} | connected so far: "
            f"{sum(q is not None for q in qeds)}"
        )

    save(
        {
            "positions": torch.cat(positions),
            "one_hot": torch.cat(one_hots),
            "qed": qeds,
            "sa": sas,
            "num_atoms": args.num_atoms,
            "config": vars(args),
        },
        args.output,
    )
    logging.info(f"Total time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == "__main__":
    main()
