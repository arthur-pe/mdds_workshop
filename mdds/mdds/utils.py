import jax
from jax import numpy as jnp
from jax import random

import equinox as eqx


def decoder_vmap(decoder, xs, neuron_ids=slice(None)):

    return jax.vmap(decoder, in_axes=(0, None), out_axes=0)(xs, neuron_ids)


@eqx.filter_jit
def reset_controls(key, controls, k, zero):

    # ===== The opt state corresponding to that control should also be reset =====
    key, subkey = random.split(key)

    def set_zero(x):

        ids = random.choice(key, x.shape[0], (k,))
        shape = [k] + list(x.shape[1:])

        x_new = random.normal(subkey, shape)/3 if not zero else jnp.zeros(shape)

        return x.at[ids].set(x_new)

    get_ys = lambda m: [m.interpolator.ys]
    new_controls = eqx.tree_at(get_ys, controls, replace_fn=set_zero)

    return new_controls


def load_model(model, load_dir, name, optimized):

    def identity(f, x):
        jnp.load(f)
        return x

    try:
        fs = {k: identity if k in optimized else eqx.default_deserialise_filter_spec for k in model.keys()}
        model = eqx.tree_deserialise_leaves(load_dir + name, model, filter_spec=fs)
        print(f'Loaded {name}')
    except:
        print(f'Initialized {name}')

    return model


def split_model(model, keys):

    return {k: model[k] for k in keys}, {k: model[k] for k in set(model.keys()) - set(keys)}
