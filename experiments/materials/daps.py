"""DAPS baseline adapted to FlowMM.

Each annealing step samples from p(x0_hat | x_t) p(y | x0_hat) with MCMC:
Metropolis-Hastings with single bit-flip proposals for the atom types and
Langevin dynamics for the continuous variables. The sample is then noised to
time t with a fresh source sample and denoised with the FlowMM ODE.
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
    log_potential,
    predict_property,
)

# max. learning rates for (frac. coords, lattice lengths, lattice angles)
LEARNING_RATES = {
    "bulk_modulus": (1e-3, 1e-2, 1e-2),
    "shear_modulus": (1e-3, 1e-2, 1e-2),
    "band_gap": (1e-5, 1e-4, 1e-4),
    "e_hull": (1e-4, 1e-3, 1e-3),
}


def parse_args():
    parser = argparse.ArgumentParser(description="DAPS material generation")
    add_prior_args(parser)
    parser.add_argument("--task", type=str, choices=PROPERTY_TASKS, required=True)
    parser.add_argument("--num-atoms", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: 125 for band_gap, otherwise 250. The loss is "
                             "averaged over the batch, so the effective step size "
                             "depends on it")
    parser.add_argument("--daps-steps", type=int, default=100)
    parser.add_argument("--ode-init-steps", type=int, default=50)
    parser.add_argument("--ode-steps", type=int, default=5)
    parser.add_argument("--mcmc-steps", type=int, default=20)
    parser.add_argument("--lr-frac", type=float, default=None)
    parser.add_argument("--lr-length", type=float, default=None)
    parser.add_argument("--lr-angle", type=float, default=None)
    parser.add_argument("--lr-min", type=float, default=1e-2,
                        help="final learning rate as a fraction of the max. learning rate")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.num_atoms is None:
        args.num_atoms = TASKS[args.task]["num_atoms"]
    if args.batch_size is None:
        args.batch_size = 125 if args.task == "band_gap" else 250
    defaults = LEARNING_RATES[args.task]
    for name, lr in zip(["lr_frac", "lr_length", "lr_angle"], defaults):
        if getattr(args, name) is None:
            setattr(args, name, lr)
    return args


def nll_and_grad(recon, prior, property_model, device, task):
    """Negative log-potential and its gradient w.r.t. the data-space sample."""
    recon = recon.detach().requires_grad_(True)
    nll = -log_potential(
        predict_property(
            *prior.to_crystals(recon), property_model, device, soft_atom_types=True
        ),
        task,
    )
    nll.mean().backward()
    return nll.detach(), recon.grad.detach()


def daps(prior, property_model, batch_size, args, device, memory_tracker):
    n = args.num_atoms
    sizes = [7 * n, 3 * n, 3, 3]

    with torch.no_grad():
        recon = prior.generate(prior.sample_source(batch_size), args.ode_init_steps)
    manifold = prior.manifold(batch_size)
    batch_indices = torch.arange(batch_size, device=recon.device)

    for step in range(args.daps_steps):
        t = torch.tensor(step / args.daps_steps, device=recon.device)
        t_batch = torch.full((batch_size, 1), t, device=recon.device)

        # learning rates decay linearly from lr to lr_min * lr
        decay = (1 - t) * (1 - args.lr_min) + args.lr_min
        lr_f = decay * args.lr_frac
        lr_l = decay * args.lr_length
        lr_t = decay * args.lr_angle

        # std of the Gaussian p(x0_hat | x_t), decreasing linearly
        r_a = 3 * (1 - t)
        r = 1 - t

        recon_a, recon_f, recon_l, recon_t = torch.split(recon, sizes, dim=1)
        recon_hat = recon.clone()

        for _ in range(args.mcmc_steps):
            nll_curr, grad_curr = nll_and_grad(
                recon_hat, prior, property_model, device, args.task
            )
            hat_a, hat_f, hat_l, hat_t = torch.split(recon_hat, sizes, dim=1)

            # Metropolis-Hastings: flip one random bit of one random atom
            prop_a = hat_a.clone()
            atom_idx = torch.randint(0, n, (batch_size,), device=recon.device)
            bit_idx = torch.randint(0, 7, (batch_size,), device=recon.device)
            prop_a[batch_indices, atom_idx * 7 + bit_idx] *= -1
            recon_prop = torch.cat([prop_a, hat_f, hat_l, hat_t], dim=1)
            nll_prop, grad_prop = nll_and_grad(
                recon_prop, prior, property_model, device, args.task
            )

            with torch.no_grad():
                log_prior_ratio = (
                    -torch.sum((prop_a - recon_a) ** 2, dim=1)
                    + torch.sum((hat_a - recon_a) ** 2, dim=1)
                ) / (2 * r_a**2)
                log_accept = log_prior_ratio + nll_curr - nll_prop
                accept = (
                    torch.rand(batch_size, device=recon.device)
                    < torch.exp(torch.clamp(log_accept, max=0))
                ).unsqueeze(1)
                hat_a = torch.where(accept, prop_a, hat_a)
                grad = torch.where(accept, grad_prop, grad_curr)
                _, grad_f, grad_l, grad_t = torch.split(grad, sizes, dim=1)

                # Langevin dynamics for the continuous variables
                hat_f = (
                    hat_f
                    - lr_f * grad_f
                    + (lr_f / r**2) * (recon_f - hat_f)
                    + torch.sqrt(2 * lr_f) * torch.randn_like(hat_f)
                )
                hat_l = (
                    hat_l
                    - lr_l * grad_l
                    + (lr_l / r**2) * (recon_l - hat_l)
                    + torch.sqrt(2 * lr_l) * torch.randn_like(hat_l)
                )
                hat_t = (
                    hat_t
                    - lr_t * grad_t
                    + (lr_t / r**2) * (recon_t - hat_t)
                    + torch.sqrt(2 * lr_t) * torch.randn_like(hat_t)
                )
                recon_hat = manifold.projx(torch.cat([hat_a, hat_f, hat_l, hat_t], dim=1))

        # noise to time t with a fresh source sample, then denoise
        with torch.no_grad():
            x0 = prior.to_base(prior.sample_source(batch_size)).to(recon.device)
            x_t, _ = manifold.cond_u(x0, recon_hat, t_batch)
            recon = prior.generate_from_xt(t, x_t, num_steps=args.ode_steps).detach()

        if step < 3:
            memory_tracker.record(f"DAPS Step {step}")

    return recon


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
        f"DAPS: {args.daps_steps} steps, {args.mcmc_steps} MCMC steps, "
        f"{args.ode_steps} ODE steps, learning rates: frac={args.lr_frac}, "
        f"length={args.lr_length}, angle={args.lr_angle}, min factor={args.lr_min}"
    )

    outputs = []
    for start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - start)
        recon = daps(prior, property_model, batch_size, args, device, memory_tracker)
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
