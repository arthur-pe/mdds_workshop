import jax
import equinox as eqx
import jax.numpy as jnp


class Integrator(eqx.Module):

    dim: int = 0

    diffusion_coef: float = None
    x0: jax.Array = None

    def __post_init__(self):

        self.x0 = jnp.zeros((self.dim,)) if self.x0 is None else self.x0

    def drift(self, *args):

        return sum(args)

    def set_initial_state(self, x0):

        self.x0 = x0

    def get_initial_state(self):

        return self.x0
