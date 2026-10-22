"""Batched elliptical slice sampling (ESS) for the source space of flow models.

Runs many independent ESS chains in parallel. Each call to the log-potential
evaluates one proposal for every chain that is still running, so chains advance
asynchronously: a chain that accepts starts its next MCMC step while others are
still shrinking their bracket. Chains that have completed all their steps are
not evaluated again.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class SampleState:
    x: torch.Tensor
    nu: torch.Tensor
    angle: torch.Tensor
    log_like: torch.Tensor
    log_thresh: torch.Tensor
    theta_min: torch.Tensor
    theta_max: torch.Tensor
    mcmc_iter: torch.Tensor
    sub_iter: torch.Tensor
    nfe: torch.Tensor


def _expand(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask.reshape(-1, *([1] * (like.dim() - 1)))


class EllipticalSliceSampler:
    """Parallel elliptical slice sampler with a standard Gaussian prior.

    Args:
        log_likelihood_fn: maps a batch of source points ``[n, ...]`` to the
            log-potential ``log g(T(z))`` of shape ``[n]``. It is only called on
            the chains that are still running, so ``n`` varies between calls.
        max_subiter: maximum number of bracket shrinkages per MCMC step. If it is
            reached, the chain stays at its current state for that step.
        logging_interval: log progress every this many sampler iterations.
    """

    def __init__(
        self,
        log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
        max_subiter: int = 100,
        logging_interval: int = 100,
    ):
        self.log_likelihood_fn = log_likelihood_fn
        self.max_subiter = max_subiter
        self.logging_interval = logging_interval

    def _log_progress(self, state: SampleState, total_steps: int, start_time: float):
        mcmc_min = state.mcmc_iter.min().item()
        mcmc_max = state.mcmc_iter.max().item()
        elapsed = time.time() - start_time
        progress = mcmc_min / total_steps
        if progress > 0:
            eta = datetime.now() + timedelta(seconds=elapsed / progress - elapsed)
        else:
            eta = "Unknown"
        logger.info(
            f"Progress: {progress * 100:.1f}% | "
            f"MCMC steps: {mcmc_min}-{mcmc_max} | "
            f"Time elapsed: {timedelta(seconds=int(elapsed))} | "
            f"ETA: {eta}"
        )

    def _initialize_states(self, x: torch.Tensor) -> SampleState:
        num_chains = x.shape[0]
        device = x.device

        nu = torch.randn_like(x)
        angle = torch.rand(num_chains, device=device) * 2 * torch.pi
        log_like = self.log_likelihood_fn(x).to(device)
        log_thresh = log_like + torch.log(torch.rand(num_chains, device=device))

        return SampleState(
            x=x,
            nu=nu,
            angle=angle,
            log_like=log_like,
            log_thresh=log_thresh,
            theta_min=angle - 2 * torch.pi,
            theta_max=angle,
            mcmc_iter=torch.zeros(num_chains, dtype=torch.long, device=device),
            sub_iter=torch.zeros(num_chains, dtype=torch.long, device=device),
            nfe=torch.ones(num_chains, dtype=torch.long, device=device),
        )

    def _proposal(self, state: SampleState, active_mask: torch.Tensor) -> torch.Tensor:
        cos_angle = _expand(torch.cos(state.angle), state.x)
        sin_angle = _expand(torch.sin(state.angle), state.x)
        proposal = state.x * cos_angle + state.nu * sin_angle
        return torch.where(_expand(active_mask, state.x), proposal, state.x)

    def _store(self, samples, state: SampleState, mask: torch.Tensor, num_warmup: int):
        """Record the current state of the ``mask`` chains for their current step."""
        is_sample = mask & (state.mcmc_iter >= num_warmup)
        if is_sample.any():
            batch_indices = is_sample.nonzero().squeeze(-1)
            sample_indices = state.mcmc_iter[batch_indices] - num_warmup
            samples[batch_indices, sample_indices] = state.x[batch_indices]

    def _start_new_mcmc_step(self, state: SampleState, mask: torch.Tensor):
        state.nu = torch.where(_expand(mask, state.nu), torch.randn_like(state.x), state.nu)
        state.angle = torch.where(
            mask, torch.rand_like(state.angle) * 2 * torch.pi, state.angle
        )
        state.log_thresh = torch.where(
            mask,
            state.log_like + torch.log(torch.rand_like(state.log_like)),
            state.log_thresh,
        )
        state.theta_min = torch.where(mask, state.angle - 2 * torch.pi, state.theta_min)
        state.theta_max = torch.where(mask, state.angle, state.theta_max)
        state.mcmc_iter = torch.where(mask, state.mcmc_iter + 1, state.mcmc_iter)
        state.sub_iter = torch.where(mask, torch.zeros_like(state.sub_iter), state.sub_iter)

    def _shrink_bracket(self, state: SampleState, reject_mask: torch.Tensor):
        state.theta_min = torch.where(
            reject_mask & (state.angle < 0), state.angle, state.theta_min
        )
        state.theta_max = torch.where(
            reject_mask & (state.angle >= 0), state.angle, state.theta_max
        )
        state.angle = torch.where(
            reject_mask,
            state.theta_min
            + torch.rand_like(state.angle) * (state.theta_max - state.theta_min),
            state.angle,
        )
        state.sub_iter = torch.where(reject_mask, state.sub_iter + 1, state.sub_iter)

    @torch.no_grad()
    def sample(
        self,
        init_x: torch.Tensor,
        num_samples: int,
        num_warmup: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ``num_warmup + num_samples`` MCMC steps for every chain.

        Args:
            init_x: initial source points, one per chain, ``[num_chains, ...]``.
            num_samples: number of post-warmup states kept per chain.
            num_warmup: number of burn-in steps discarded per chain.

        Returns:
            samples: ``[num_chains, num_samples, ...]`` post-warmup states.
            nfe: ``[num_chains]`` number of log-potential evaluations per chain.
        """
        num_chains = init_x.shape[0]
        total_steps = num_samples + num_warmup
        start_time = time.time()

        samples = (
            init_x.unsqueeze(1)
            .expand(-1, num_samples, *([-1] * (init_x.dim() - 1)))
            .clone()
        )
        state = self._initialize_states(init_x)

        logger.info(
            f"Starting sampling with {num_chains} chains, "
            f"total steps {total_steps} (warmup: {num_warmup})"
        )

        iteration = 0
        while True:
            active_mask = state.mcmc_iter < total_steps
            if not active_mask.any():
                break

            iteration += 1
            if iteration % self.logging_interval == 0:
                self._log_progress(state, total_steps, start_time)

            x_prop = self._proposal(state, active_mask)

            # only evaluate chains that are still running
            active_indices = active_mask.nonzero().squeeze(-1)
            log_like_prop = torch.full_like(state.log_like, float("-inf"))
            log_like_prop[active_indices] = self.log_likelihood_fn(
                x_prop[active_indices]
            ).to(log_like_prop.device)
            state.nfe = torch.where(active_mask, state.nfe + 1, state.nfe)

            accept_mask = (log_like_prop > state.log_thresh) & active_mask
            reject_mask = ~accept_mask & active_mask

            if accept_mask.any():
                state.x = torch.where(_expand(accept_mask, state.x), x_prop, state.x)
                state.log_like = torch.where(accept_mask, log_like_prop, state.log_like)
                self._store(samples, state, accept_mask, num_warmup)
                self._start_new_mcmc_step(state, accept_mask)

            if reject_mask.any():
                self._shrink_bracket(state, reject_mask)
                # give up on this ellipse: the chain stays at its current state
                stuck_mask = reject_mask & (state.sub_iter >= self.max_subiter)
                if stuck_mask.any():
                    self._store(samples, state, stuck_mask, num_warmup)
                    self._start_new_mcmc_step(state, stuck_mask)

        logger.info("Sampling completed!")
        self._log_progress(state, total_steps, start_time)
        return samples, state.nfe
