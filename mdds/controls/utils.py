from .interpolated_controls import LinearControl, LowRankLinearInterpolator, LinearInterpolator

from mdds.controls.parameterization import resolve_parameterization

import equinox as eqx

from jax import numpy as jnp
import jax


def get_grid_hypersphere(trial_dim, manifold_dim):

    theta = jnp.linspace(0, 2*jnp.pi, trial_dim)

    temp =  jnp.stack([jnp.cos(theta), jnp.sin(theta)]+[jnp.zeros((len(theta),)) for i in range(manifold_dim-2)], axis=-1)

    return temp #/ jnp.linalg.norm(temp, axis=-1, ord=0.1)[..., jnp.newaxis]


def build_low_rank_control(ts, dim, trial_dim, rank, key=None):

    return LinearControl(jnp.tile(ts, (trial_dim, 1)), LowRankLinearInterpolator,
                         trial_dim=trial_dim, dim=dim, rank=rank, key=key)


def build_control(ts, grid, key=None):

    return LinearControl(jnp.tile(ts, (grid.shape[0], 1)), LinearInterpolator,
                         dim=grid.shape[-1], trial_dim=grid.shape[0], init_coef=grid[:, jnp.newaxis], key=key)


def evaluate_control(controls, ts):

    controls = resolve_parameterization(controls)

    def f(c, t): return jax.vmap(c)(t)

    temp = eqx.filter_vmap(f, in_axes=(eqx.if_array(0), 0), out_axes=0)(controls, ts)

    return temp


def batch_controls(controls, idx):

    def get_arrays(x):
        return x[idx]

    parameterized_model = controls
    parameterized_model = eqx.tree_at(lambda x: x.interpolator.ys, parameterized_model, replace_fn=get_arrays)
    #parameterized_model = eqx.tree_at(lambda x: x.interpolator.init_coef, parameterized_model, replace_fn=get_arrays)
    parameterized_model = eqx.tree_at(lambda x: x.interpolator.ts, parameterized_model, replace_fn=get_arrays)
    #parameterized_model = eqx.tree_at(lambda x: x.ts, parameterized_model, replace_fn=get_arrays)
    #parameterized_model = eqx.tree_at(lambda x: x.init_coef, parameterized_model, replace_fn=get_arrays)
    parameterized_model = eqx.tree_at(lambda x: x.t0, parameterized_model, replace_fn=get_arrays)
    parameterized_model = eqx.tree_at(lambda x: x.t1, parameterized_model, replace_fn=get_arrays)

    return parameterized_model