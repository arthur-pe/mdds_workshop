import jax
import equinox as eqx
import jax.numpy as jnp


class OU(eqx.Module):

    dim: int = 0

    mean: jax.Array = None
    drift_coef: float = None
    diffusion_coef: float = None
    x0: jax.Array = None

    def __post_init__(self):

        self.mean = jnp.zeros(self.dim) if self.mean is None else self.mean
        self.drift_coef = 1.0 if self.drift_coef is None else self.drift_coef
        self.diffusion_coef = 0.5 if self.diffusion_coef is None else self.diffusion_coef
        self.x0 = self.mean if self.x0 is None else self.x0

    def drift(self, x, *args):

        return self.drift_coef*(self.mean - x)

    def diffusion(self, x, *args):

        return self.diffusion_coef

    def get_initial_state(self):

        return self.x0
