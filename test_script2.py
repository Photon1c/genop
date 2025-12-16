"""
Experiment script 2: "excite -> maintain" controller.

This is built directly off:
- `generative_optogenetics/test_script.py` (environment + baselines)
- `generative_optogenetics/implementation.md` (analysis + next-step ideas)

Key idea:
- Use the greedy controller early to quickly match the phenotype (controllability)
- Then switch to a maintenance regime that:
  - pulses sparsely (duty cycle) instead of continuous blasting
  - tapers based on remaining toxicity budget
  - avoids overshoot by only correcting deficits (target - expr)+ with a margin
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    # When run from repo root: python generative_optogenetics/test_script2.py
    from generative_optogenetics.test_script import (
        PetriDishEnv,
        controller_greedy,
        controller_random,
    )
except ModuleNotFoundError:
    # When run from inside the folder: python test_script2.py
    from test_script import PetriDishEnv, controller_greedy, controller_random  # type: ignore


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_info: Dict[str, float]
    history: Dict[str, List[float]]


class DesensitizingPetriDishEnv(PetriDishEnv):
    """
    PetriDishEnv + opsin desensitization.

    Mechanism:
    - Maintain a per-cell sensitivity state `sens` in [0,1], initialized to 1.
    - Sensitivity decreases with cumulative activation (light * opsin), and recovers slowly.
    - Effective opsin becomes: opsin_eff = opsin * sens.

    This forces genuine pulse scheduling: continuous illumination reduces responsiveness over time.
    """

    def __init__(
        self,
        *args,
        desense_k: float = 1.25,
        recover_k: float = 0.06,
        sens_floor: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.desense_k = float(desense_k)
        self.recover_k = float(recover_k)
        self.sens_floor = float(np.clip(sens_floor, 0.0, 1.0))
        # `reset()` already called in super().__init__(), so ensure sens exists.
        if not hasattr(self, "sens"):
            self.sens = np.ones((self.h, self.w), dtype=np.float32)

    def reset(self):
        obs = super().reset()
        self.sens = np.ones((self.h, self.w), dtype=np.float32)
        # Time-integrated error (Integral of MSE over time, discrete dt sum).
        self.ie = 0.0
        return obs

    def step(self, light):
        light = np.clip(light, 0.0, 1.0)

        # Desensitization: cumulative activation reduces sensitivity, with slow recovery.
        # Activation uses raw light * opsin (pre-desense) to reflect "dose history".
        activation = (light * self.opsin).astype(np.float32)
        d_sens = (-self.desense_k * activation + self.recover_k * (1.0 - self.sens)) * self.dt
        self.sens = np.clip(self.sens + d_sens, self.sens_floor, 1.0).astype(np.float32)

        # Apply sensitivity to opsin response.
        opsin_eff = (self.opsin * self.sens).astype(np.float32)

        # Saturating opsin response (same functional form as base env)
        drive = (light * opsin_eff) / (self.sat + (light * opsin_eff) + 1e-8)

        # Expression dynamics: rise with drive, decay to baseline 0
        d_expr = (self.k_on * drive - self.k_off * self.expr) * self.dt

        # Diffusion/coupling
        d_expr += self.diff * self._laplacian(self.expr) * self.dt

        self.expr = np.clip(self.expr + d_expr, 0.0, 1.0)

        # Phototoxicity: cumulative light dose hurts health (unchanged)
        self.health = np.clip(self.health - (self.tox_k * light * self.dt), 0.0, 1.0)

        # Costs / reward
        mse = np.mean((self.expr - self.target) ** 2)
        energy = np.mean(light)
        tox = np.mean(1.0 - self.health)

        # Time-integrated error objective (explicitly values maintenance):
        # reward is the negative incremental integral at this step.
        self.ie += float(mse) * float(self.dt)
        reward = -(
            float(mse) * float(self.dt)
            + self.energy_weight * float(energy) * float(self.dt)
            + self.tox_weight * float(tox) * float(self.dt)
        )

        self.t += 1
        done = (self.t >= self.max_steps) or (np.min(self.health) <= 0.0)

        info = {
            "mse": float(mse),
            "energy": float(energy),
            "tox": float(tox),
            "sens": float(np.mean(self.sens)),
            "ie": float(self.ie),
        }
        return self.observe(), float(reward), bool(done), info


def _cap_mean_light(light: np.ndarray, mean_cap: float) -> np.ndarray:
    """Scale down a light field so its mean <= mean_cap (keeps shape, preserves [0,1])."""
    mean_cap = float(np.clip(mean_cap, 0.0, 1.0))
    m = float(np.mean(light))
    if m <= 1e-12 or m <= mean_cap:
        return np.clip(light, 0.0, 1.0)
    scale = mean_cap / m
    return np.clip(light * scale, 0.0, 1.0)


def _toxicity_mean_light_cap(env: PetriDishEnv, tox_budget: float, remaining_steps: int) -> float:
    """
    Convert a remaining toxicity budget into a per-step *mean light* cap.

    In `PetriDishEnv.step`, toxicity evolves as:
      health -= tox_k * light * dt
    so mean(1-health) increases approximately:
      d tox ~= tox_k * mean(light) * dt
    """
    remaining_steps = max(int(remaining_steps), 1)
    tox_budget = float(np.clip(tox_budget, 0.0, 1.0))
    current_tox = float(np.mean(1.0 - env.health))
    remaining_tox = max(0.0, tox_budget - current_tox)

    denom = float(env.tox_k * env.dt * remaining_steps) + 1e-12
    mean_cap = remaining_tox / denom
    return float(np.clip(mean_cap, 0.0, 1.0))


@dataclass(frozen=True)
class LatentParams:
    """
    4D latent controller:
    - pulse_period: int >= 1
    - duty_cycle: fraction of period where light is ON, in [0,1]
    - amplitude: scalar multiplier for normalized spatial light field, in [0,1]
    - decay_threshold: if MSE is below this, we go dark (explicit maintenance objective), in [0,1]
    """

    pulse_period: int
    duty_cycle: float
    amplitude: float
    decay_threshold: float


def _decode_latent(x: np.ndarray) -> LatentParams:
    """
    Map an unconstrained CMA-ES vector to valid controller parameters.
    This keeps CMA-ES continuous, while the controller sees constrained values.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 4:
        raise ValueError("latent vector must have at least 4 dims")

    # Pulse period: map to [1, 20] and round.
    period = int(np.clip(np.round(x[0]), 1, 20))

    # Duty cycle, amplitude, decay threshold: clip to [0,1]
    duty = float(np.clip(x[1], 0.0, 1.0))
    amp = float(np.clip(x[2], 0.0, 1.0))
    thr = float(np.clip(x[3], 0.0, 1.0))
    return LatentParams(pulse_period=period, duty_cycle=duty, amplitude=amp, decay_threshold=thr)


def controller_latent(env: PetriDishEnv, params: LatentParams, *, tox_budget: float = 0.22) -> np.ndarray:
    """
    Latent controller (3–5 params) that:
    - pulses globally with (period, duty)
    - scales overall intensity by amplitude
    - uses a decay threshold on instantaneous MSE to go dark (values maintenance)
    - still budgets toxicity over the remaining horizon (keeps it stable)
    """
    mse = float(np.mean((env.expr - env.target) ** 2))
    if mse < params.decay_threshold:
        return np.zeros((env.h, env.w), dtype=np.float32)

    # Global pulse gate
    on_steps = int(np.clip(np.round(params.duty_cycle * params.pulse_period), 0, params.pulse_period))
    phase = int(env.t) % int(params.pulse_period)
    if phase >= on_steps:
        return np.zeros((env.h, env.w), dtype=np.float32)

    # Spatial pattern: deficit-only correction (target - expr)+ weighted by opsin + health.
    deficit = np.clip(env.target - env.expr, 0.0, 1.0)
    raw = (deficit * env.opsin * env.health).astype(np.float32)
    mx = float(raw.max()) + 1e-12
    light = np.clip(raw / mx, 0.0, 1.0)

    # Apply latent amplitude
    light = np.clip(light * float(params.amplitude), 0.0, 1.0).astype(np.float32)

    # Toxicity budgeting (mean cap)
    remaining_steps = env.max_steps - env.t
    mean_cap = _toxicity_mean_light_cap(env, tox_budget=tox_budget, remaining_steps=remaining_steps)
    return _cap_mean_light(light, mean_cap).astype(np.float32)


class CMAES:
    """
    Minimal CMA-ES implementation (dependency-free) for small parameter counts.
    Minimizes objective f(x).
    """

    def __init__(self, x0: np.ndarray, sigma0: float, popsize: Optional[int] = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n = int(np.asarray(x0).size)
        self.mean = np.asarray(x0, dtype=np.float64).reshape(self.n)
        self.sigma = float(sigma0)

        self.popsize = int(popsize) if popsize is not None else int(4 + np.floor(3 * np.log(self.n)))
        self.popsize = max(self.popsize, 6)
        self.mu = self.popsize // 2

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = float(np.sum(self.weights) ** 2 / np.sum(self.weights**2))

        # Strategy parameter settings: adaptation
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

        self.pc = np.zeros(self.n, dtype=np.float64)
        self.ps = np.zeros(self.n, dtype=np.float64)
        self.C = np.eye(self.n, dtype=np.float64)
        self.B = np.eye(self.n, dtype=np.float64)
        self.D = np.ones(self.n, dtype=np.float64)
        self.invsqrtC = np.eye(self.n, dtype=np.float64)

        self.chiN = float(np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n**2)))
        self._eigeneval = 0
        self._counteval = 0

    def ask(self) -> np.ndarray:
        self._maybe_update_eigendecomp()
        z = self.rng.normal(0.0, 1.0, size=(self.popsize, self.n))
        y = z @ (self.B * self.D).T
        x = self.mean + self.sigma * y
        return x

    def tell(self, X: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, float]:
        """Update state given evaluated population. Returns (best_x, best_f)."""
        X = np.asarray(X, dtype=np.float64)
        fitness = np.asarray(fitness, dtype=np.float64).reshape(-1)
        self._counteval += int(fitness.size)

        idx = np.argsort(fitness)
        Xs = X[idx]
        fs = fitness[idx]

        xold = self.mean.copy()
        Xmu = Xs[: self.mu]
        self.mean = np.sum(Xmu * self.weights[:, None], axis=0)

        y = (self.mean - xold) / (self.sigma + 1e-12)
        self._maybe_update_eigendecomp()

        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ y)
        norm_ps = float(np.linalg.norm(self.ps))
        hsig = float(
            norm_ps / np.sqrt(1 - (1 - self.cs) ** (2 * self._counteval / self.popsize)) / self.chiN
            < (1.4 + 2 / (self.n + 1))
        )
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        # Rank-one + rank-mu update of covariance
        artmp = (Xmu - xold) / (self.sigma + 1e-12)
        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + delta_hsig * self.C)
            + self.cmu * (artmp.T @ (np.diag(self.weights) @ artmp))
        )

        # Step-size control
        self.sigma *= float(np.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1)))

        return Xs[0].copy(), float(fs[0])

    def _maybe_update_eigendecomp(self) -> None:
        # Recompute eigendecomposition occasionally
        if self._counteval - self._eigeneval < self.popsize / (self.c1 + self.cmu) / self.n / 10:
            return
        self._eigeneval = self._counteval
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
        D2, B = np.linalg.eigh(self.C)
        D2 = np.maximum(D2, 1e-20)
        self.D = np.sqrt(D2)
        self.B = B
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T


def evaluate_latent(
    x: np.ndarray,
    *,
    seeds: List[int],
    h: int = 48,
    w: int = 48,
    tox_budget: float = 0.22,
    verbose: bool = False,
) -> float:
    """
    Objective for CMA-ES: minimize -avg_total_reward across seeds.
    Uses the desensitizing environment and time-integrated reward (already in env).
    """
    params = _decode_latent(x)
    totals: List[float] = []
    for sd in seeds:
        env = DesensitizingPetriDishEnv(h=h, w=w, seed=sd)
        res = run_episode(
            env,
            lambda e: controller_latent(e, params, tox_budget=tox_budget),
            verbose_every=999999 if not verbose else 25,
            record=False,
        )
        totals.append(res.total_reward)
    return -float(np.mean(totals))


def optimize_latent_cmaes(
    *,
    iters: int,
    popsize: int,
    seed: int,
    eval_seeds: List[int],
    sigma0: float = 0.6,
) -> Tuple[LatentParams, np.ndarray, float]:
    # Reasonable starting point in the decoded space:
    # period ~ 6, duty ~ 0.33, amp ~ 0.5, thr ~ 0.02
    x0 = np.array([6.0, 0.33, 0.50, 0.02], dtype=np.float64)
    es = CMAES(x0=x0, sigma0=sigma0, popsize=popsize, seed=seed)

    best_x = x0.copy()
    best_f = float("inf")
    for _ in range(int(iters)):
        X = es.ask()
        f = np.array([evaluate_latent(xi, seeds=eval_seeds) for xi in X], dtype=np.float64)
        gen_best_x, gen_best_f = es.tell(X, f)
        if gen_best_f < best_f:
            best_f = gen_best_f
            best_x = gen_best_x

    return _decode_latent(best_x), best_x, best_f


def run_episode(
    env: PetriDishEnv,
    policy_fn: Callable[[PetriDishEnv], np.ndarray],
    *,
    verbose_every: int = 25,
    record: bool = True,
) -> EpisodeResult:
    env.reset()
    total = 0.0
    history: Dict[str, List[float]] = {"reward": [], "mse": [], "energy": [], "tox": [], "sens": [], "ie": []}

    final_info: Dict[str, float] = {"mse": float("nan"), "energy": float("nan"), "tox": float("nan")}
    for t in range(env.max_steps):
        light = policy_fn(env)
        _, r, done, info = env.step(light)
        total += float(r)
        final_info = info

        if record:
            history["reward"].append(float(r))
            history["mse"].append(float(info["mse"]))
            history["energy"].append(float(info["energy"]))
            history["tox"].append(float(info["tox"]))
            history["sens"].append(float(info.get("sens", float("nan"))))
            history["ie"].append(float(info.get("ie", float("nan"))))

        if (t % verbose_every) == 0:
            print(
                f"t={t:03d}  r={r:+.4f}  mse={info['mse']:.5f}  "
                f"energy={info['energy']:.3f}  tox={info['tox']:.3f}"
                + (f"  sens={info['sens']:.3f}" if "sens" in info else "")
                + (f"  ie={info['ie']:.4f}" if "ie" in info else "")
            )
        if done:
            break

    print(
        f"episode done: steps={env.t}, total_reward={total:.2f}, "
        f"final_mse={final_info['mse']:.5f}, tox={final_info['tox']:.3f}"
        + (f", ie={final_info['ie']:.4f}" if "ie" in final_info else "")
    )
    return EpisodeResult(total_reward=float(total), final_info=final_info, history=history)


def _summarize(name: str, res: EpisodeResult) -> None:
    h = res.history
    if not h["mse"]:
        print(f"{name}: (no history)")
        return
    best_mse = float(np.min(h["mse"]))
    final_mse = float(h["mse"][-1])
    final_tox = float(h["tox"][-1])
    mean_energy = float(np.mean(h["energy"]))
    sens_series = [x for x in h.get("sens", []) if not np.isnan(x)]
    final_sens = float(sens_series[-1]) if sens_series else float("nan")
    ie_series = [x for x in h.get("ie", []) if not np.isnan(x)]
    final_ie = float(ie_series[-1]) if ie_series else float("nan")
    print(
        f"{name}: total={res.total_reward:+.2f} | best_mse={best_mse:.5f} | "
        f"final_mse={final_mse:.5f} | mean_energy={mean_energy:.3f} | final_tox={final_tox:.3f}"
        + (f" | final_sens={final_sens:.3f}" if not np.isnan(final_sens) else "")
        + (f" | final_ie={final_ie:.4f}" if not np.isnan(final_ie) else "")
    )


def main() -> None:
    args = argparse.ArgumentParser(description="Generative optogenetics: latent controller + CMA-ES")
    args.add_argument("--optimize", action="store_true", help="Run CMA-ES to tune latent controller params")
    args.add_argument("--iters", type=int, default=25, help="CMA-ES generations")
    args.add_argument("--popsize", type=int, default=10, help="CMA-ES population size")
    args.add_argument("--seed", type=int, default=0, help="CMA-ES RNG seed")
    args.add_argument("--eval-seeds", type=str, default="1,2,3", help="Comma-separated env seeds for evaluation")
    ns = args.parse_args()

    eval_seeds = [int(s.strip()) for s in ns.eval_seeds.split(",") if s.strip()]

    # Use desensitization by default to enforce real pulse scheduling.
    env = DesensitizingPetriDishEnv(h=48, w=48, seed=eval_seeds[0] if eval_seeds else 1)

    print("\n--- Greedy baseline ---")
    greedy_res = run_episode(env, controller_greedy, verbose_every=25)

    print("\n--- Random baseline ---")
    random_res = run_episode(env, lambda e: controller_random(e, max_gaussians=3), verbose_every=25)

    best_params: Optional[LatentParams] = None
    if ns.optimize:
        print("\n--- CMA-ES optimizing latent controller ---")
        best_params, _, best_f = optimize_latent_cmaes(
            iters=ns.iters,
            popsize=ns.popsize,
            seed=ns.seed,
            eval_seeds=eval_seeds,
        )
        print(
            "best latent params: "
            f"period={best_params.pulse_period}, duty={best_params.duty_cycle:.3f}, "
            f"amp={best_params.amplitude:.3f}, thr={best_params.decay_threshold:.4f} "
            f"(objective={best_f:.4f})"
        )
    else:
        best_params = _decode_latent(np.array([6.0, 0.33, 0.50, 0.02], dtype=np.float64))

    print("\n--- Latent controller (params) ---")
    latent_res = run_episode(env, lambda e: controller_latent(e, best_params), verbose_every=25)

    print("\n--- Summary ---")
    _summarize("greedy", greedy_res)
    _summarize("random", random_res)
    _summarize("latent", latent_res)


if __name__ == "__main__":
    main()

