import numpy as np

def gaussian2d(h, w, cx, cy, sx, sy):
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))

class PetriDishEnv:
    """
    Minimalist 'petri dish' optogenetics sandbox.

    State:
      expr:    gene expression proxy in [0, 1] (float grid)
      opsin:   light sensitivity in [0, 1] (float grid)
      health:  phototoxicity budget in [0, 1] (float grid)

    Action:
      light: intensity field in [0, 1] (float grid) OR low-dim params -> field

    Reward:
      -MSE(expr, target) - energy_cost - toxicity_cost
    """
    def __init__(self, h=48, w=48, dt=0.1, seed=0):
        self.h, self.w = h, w
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Dynamics knobs (toy but interpretable)
        self.k_on = 2.0         # light->expression gain
        self.k_off = 0.8        # decay back to baseline
        self.sat = 0.7          # saturating nonlinearity
        self.diff = 0.15        # diffusion strength
        self.tox_k = 0.25       # phototoxicity rate
        self.max_steps = 200

        # Costs
        self.energy_weight = 0.06
        self.tox_weight = 0.4

        self.reset()

    def reset(self):
        # opsin: spatial heterogeneity
        self.opsin = np.clip(self.rng.normal(0.7, 0.15, (self.h, self.w)), 0.2, 1.0)

        # expression starts low + noise
        self.expr = np.clip(self.rng.normal(0.05, 0.02, (self.h, self.w)), 0.0, 0.2)

        self.health = np.ones((self.h, self.w), dtype=np.float32)
        self.t = 0
        self.target = self.make_target(kind="ring")
        return self.observe()

    def make_target(self, kind="ring"):
        if kind == "left_half":
            tgt = np.zeros((self.h, self.w), dtype=np.float32)
            tgt[:, : self.w // 2] = 1.0
            return tgt
        if kind == "ring":
            y = np.linspace(-1, 1, self.h)[:, None]
            x = np.linspace(-1, 1, self.w)[None, :]
            r = np.sqrt(x * x + y * y)
            tgt = np.exp(-((r - 0.55) ** 2) / (2 * 0.07 ** 2))
            return np.clip(tgt, 0, 1).astype(np.float32)
        if kind == "spot":
            return gaussian2d(self.h, self.w, self.w*0.5, self.h*0.5, 4.5, 4.5).astype(np.float32)
        raise ValueError("unknown target kind")

    def observe(self, noise=0.01, partial=False):
        obs = self.expr.copy()
        if partial:
            # crude partial observability: blur by downsampling/upsampling
            small = obs[::3, ::3]
            obs = np.kron(small, np.ones((3, 3)))[: self.h, : self.w]
        obs = obs + self.rng.normal(0.0, noise, obs.shape)
        return np.clip(obs, 0, 1)

    def _laplacian(self, x):
        # 5-point stencil with wrap-free boundaries
        up = np.roll(x, -1, axis=0); up[-1, :] = x[-1, :]
        dn = np.roll(x,  1, axis=0); dn[ 0, :] = x[ 0, :]
        lt = np.roll(x, -1, axis=1); lt[:, -1] = x[:, -1]
        rt = np.roll(x,  1, axis=1); rt[:,  0] = x[:,  0]
        return (up + dn + lt + rt - 4.0 * x)

    def step(self, light):
        light = np.clip(light, 0.0, 1.0)

        # Saturating opsin response
        drive = (light * self.opsin) / (self.sat + (light * self.opsin) + 1e-8)

        # Expression dynamics: rise with drive, decay to baseline 0
        d_expr = (self.k_on * drive - self.k_off * self.expr) * self.dt

        # Diffusion/coupling
        d_expr += self.diff * self._laplacian(self.expr) * self.dt

        self.expr = np.clip(self.expr + d_expr, 0.0, 1.0)

        # Phototoxicity: cumulative light dose hurts health
        self.health = np.clip(self.health - (self.tox_k * light * self.dt), 0.0, 1.0)

        # Costs / reward
        mse = np.mean((self.expr - self.target) ** 2)
        energy = np.mean(light)
        tox = np.mean(1.0 - self.health)

        reward = -(mse + self.energy_weight * energy + self.tox_weight * tox)

        self.t += 1
        done = (self.t >= self.max_steps) or (np.min(self.health) <= 0.0)

        info = {"mse": float(mse), "energy": float(energy), "tox": float(tox)}
        return self.observe(), float(reward), bool(done), info


# ----------------------------
# Baseline controllers
# ----------------------------

def controller_random(env, max_gaussians=3):
    """Random low-dim generator: sum of a few Gaussians."""
    h, w = env.h, env.w
    light = np.zeros((h, w), dtype=np.float32)
    for _ in range(max_gaussians):
        cx = env.rng.uniform(0, w)
        cy = env.rng.uniform(0, h)
        sx = env.rng.uniform(2.0, 10.0)
        sy = env.rng.uniform(2.0, 10.0)
        amp = env.rng.uniform(0.2, 1.0)
        light += amp * gaussian2d(h, w, cx, cy, sx, sy)
    return np.clip(light, 0, 1)

def controller_greedy(env):
    """
    Greedy heuristic: illuminate where target > current,
    scaled by opsin and remaining health.
    """
    delta = np.clip(env.target - env.expr, 0, 1)
    light = delta * env.opsin * env.health
    # normalize to [0,1]
    m = light.max() + 1e-8
    return np.clip(light / m, 0, 1)

def run_episode(env, policy_fn, verbose_every=25):
    obs = env.reset()
    total = 0.0
    for t in range(env.max_steps):
        light = policy_fn(env)
        obs, r, done, info = env.step(light)
        total += r
        if (t % verbose_every) == 0:
            print(f"t={t:03d}  reward={r:+.4f}  mse={info['mse']:.4f}  energy={info['energy']:.3f}  tox={info['tox']:.3f}")
        if done:
            break
    print(f"episode done: steps={env.t}, total_reward={total:.2f}, final_mse={info['mse']:.4f}, tox={info['tox']:.3f}")
    return total, info

if __name__ == "__main__":
    env = PetriDishEnv(h=48, w=48, seed=1)

    print("\n--- Greedy baseline ---")
    run_episode(env, controller_greedy)

    print("\n--- Random baseline ---")
    run_episode(env, lambda e: controller_random(e, max_gaussians=3))
