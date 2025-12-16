# Generative Optogenetics: Latent Controller + CMA-ES
## Evolving AI Controller To Be Upgraded Soon

A minimal sandbox for **generative optogenetic control** with opsin desensitization, time-integrated error objectives, and evolutionary optimization.

## Overview

This project implements a toy "petri dish" environment where:
- **State**: Gene expression (`expr`), opsin sensitivity (`opsin`), cell health (`health`)
- **Action**: Spatial light field (`light`) that drives expression via opsin activation
- **Dynamics**: Expression rises with light, decays naturally, diffuses spatially, and accumulates phototoxicity
- **Objective**: Minimize time-integrated MSE between expression and target pattern, while managing energy and toxicity

### Key Features

1. **Opsin Desensitization**: Response decreases with cumulative activation, forcing genuine pulse scheduling
2. **Time-Integrated Error**: Reward explicitly values maintenance (area under error curve)
3. **Latent Controller**: 4-parameter policy (pulse period, duty cycle, amplitude, decay threshold)
4. **CMA-ES Optimization**: Evolutionary search for optimal controller parameters

## Quick Start

### Basic Usage

Run with default (hand-tuned) latent controller parameters:

```bash
python generative_optogenetics/test_script2.py
```

This compares:
- **Greedy baseline**: Illuminates where `target > expr`, scaled by opsin and health
- **Random baseline**: Random Gaussian light fields
- **Latent controller**: Parameterized pulse-based policy

### Optimize with CMA-ES

Find optimal controller parameters:

```bash
python generative_optogenetics/test_script2.py --optimize --iters 25 --popsize 10
```

Options:
- `--optimize`: Enable CMA-ES optimization
- `--iters N`: Number of CMA-ES generations (default: 25)
- `--popsize N`: Population size per generation (default: 10)
- `--seed N`: RNG seed for CMA-ES (default: 0)
- `--eval-seeds "1,2,3"`: Comma-separated environment seeds for evaluation (default: "1,2,3")

## Architecture

### Environment: `DesensitizingPetriDishEnv`

Extends `PetriDishEnv` from `test_script.py` with:

- **Sensitivity state** `sens ∈ [sens_floor, 1]`:
  - Decreases with cumulative activation: `d_sens = (-desense_k * activation + recover_k * (1 - sens)) * dt`
  - Effective opsin: `opsin_eff = opsin * sens`
  - Forces pulse scheduling: continuous light → burnout → reduced responsiveness

- **Time-integrated error**:
  - Tracks `ie = Σ(mse * dt)` over episode
  - Reward: `-(mse*dt + energy_weight*energy*dt + tox_weight*tox*dt)`
  - Explicitly values maintenance (low error over time)

### Controller: `controller_latent`

4-parameter policy:

1. **`pulse_period`** (int ≥ 1): Global pulse period (e.g., 6 steps)
2. **`duty_cycle`** (float ∈ [0,1]): Fraction of period where light is ON
3. **`amplitude`** (float ∈ [0,1]): Scalar multiplier for normalized spatial light field
4. **`decay_threshold`** (float ∈ [0,1]): If MSE < threshold, go dark (maintenance mode)

**Behavior**:
- If `mse < decay_threshold`: return zero light (maintenance)
- Else: pulse with `(period, duty_cycle)`, scale by `amplitude`
- Spatial pattern: deficit-only correction `(target - expr)+` weighted by `opsin * health`
- Toxicity budgeting: caps mean light to stay within budget over remaining horizon

### Optimization: CMA-ES

Self-contained implementation (no external dependencies):
- Minimizes `-avg_total_reward` across multiple environment seeds
- Maps unconstrained search space to valid parameter ranges
- Starting point: `[period=6, duty=0.33, amp=0.50, thr=0.02]`

## Results Interpretation

### Metrics

- **`total_reward`**: Sum of per-step rewards (higher = better)
- **`best_mse`**: Minimum MSE during episode (convergence quality)
- **`final_mse`**: Final MSE (maintenance quality)
- **`final_ie`**: Time-integrated error (area under MSE curve; lower = better)
- **`mean_energy`**: Average light intensity (efficiency)
- **`final_tox`**: Final toxicity (safety)
- **`final_sens`**: Final mean sensitivity (desensitization impact)

### Expected Behavior

**Greedy baseline**:
- ✅ Fast convergence (low `best_mse`)
- ❌ Poor maintenance (high `final_mse`, `final_ie`)
- ❌ High toxicity, desensitization

**Random baseline**:
- ❌ No convergence
- ❌ High energy, toxicity, desensitization

**Latent controller (optimized)**:
- ✅ Moderate convergence
- ✅ Better maintenance (lower `final_ie` than random)
- ✅ Low energy, toxicity
- ✅ Preserves sensitivity (pulsing allows recovery)

## Files

- **`test_script.py`**: Base environment + greedy/random baselines
- **`test_script2.py`**: Desensitization + latent controller + CMA-ES
- **`implementation.md`**: Analysis of greedy vs random results, design rationale

## Dependencies

- `numpy` (only external dependency)
- Python 3.8+

## Design Philosophy

This sandbox validates a core hypothesis from `implementation.md`:

> **Optogenetics is not a static control problem. It is a dose–timing–maintenance problem.**

The greedy controller proves **controllability** (target is reachable) but fails at **stability** (cannot hold pattern sustainably). The latent controller + CMA-ES learns to:
- Front-load excitation (fast alignment)
- Switch regimes (excite → maintain)
- Pulse sparsely (exploit recovery windows)
- Budget toxicity over horizon

This creates a **non-trivial but learnable** control problem where AI-driven policies are structurally necessary.

## Future Directions

- Add delayed opsin kinetics (realistic temporal dynamics)
- Introduce spatial "maintenance halos" (diffusion-based control)
- Wire into PPO/other RL algorithms (learned policies vs evolutionary search)
- Multi-objective optimization (Pareto fronts for energy vs error vs toxicity)

