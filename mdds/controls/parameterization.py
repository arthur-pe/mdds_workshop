import jax
from jax import numpy as jnp
import equinox as eqx


class Parameterization(eqx.Module):#, jax.Array

    #W: jax.Array = None

    def get(self) -> jax.Array:

        pass


def is_parameterized(x):

    return isinstance(x, Parameterization)


def get_parameterization(x):

    return x.get() if is_parameterized(x) else x


def resolve_parameterization(model):

    get_weights = lambda m: [x
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_parameterized)
                             if is_parameterized(x)]

    parameterized_model = eqx.tree_at(get_weights, model, replace_fn=get_parameterization)

    return parameterized_model
