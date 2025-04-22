from mdds.controls.parameterization import Parameterization

from diffrax import AbstractPath, LinearInterpolation, CubicInterpolation, backward_hermite_coefficients

from jax import random
from jax import numpy as jnp

import equinox as eqx
import jax

from typing import Union

import numpy as np


class WrapperTupleControl(AbstractPath):

    ts: jax.Array = eqx.field(static=True)

    t0: float = None
    t1: float = None

    controls: tuple

    def __init__(self, controls):

        self.controls = controls

        self.ts = controls[0].ts

        self.t0, self.t1 = controls[0].t0, controls[0].t1

    def evaluate(self, t0, t1=None, left: bool = True):
        return tuple([c.evaluate(t0=t0, t1=t1, left=left) if c is not None else None for c in self.controls])

    def __call__(self, ts):
        return self.evaluate(ts)


class Interpolator(Parameterization):

    ts: jax.Array = eqx.field(static=False)
    dim: int
    trial_dim: int
    key: random.PRNGKey = None

    def get_ys(self):

        raise NotImplementedError('Implement get_ys method.')

    def get(self):

        ys = self.get_ys()

        #coefs = eqx.filter_vmap(backward_hermite_coefficients, in_axes=(0, 0), out_axes=0)(jax.lax.stop_gradient(self.ts), ys)
        #return eqx.filter_vmap(CubicInterpolation, in_axes=(0, 0), out_axes=0)(jax.lax.stop_gradient(self.ts), coefs)

        return eqx.filter_vmap(LinearInterpolation, in_axes=(0, 0), out_axes=0)(jax.lax.stop_gradient(self.ts), ys)


class LinearInterpolator(Interpolator):

    init_coef: jax.Array = eqx.field(default=None, static=False)

    ys: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        if self.key is None:
            self.ys = jnp.ones((self.trial_dim, self.ts.shape[1], self.dim))*self.init_coef
        else:
            key, subkey = random.split(self.key)
            self.ys = random.normal(subkey, (self.trial_dim, self.ts.shape[1], self.dim))*self.init_coef

    def get_ys(self):

        return self.ys


class LowRankLinearInterpolator(Interpolator):

    rank: int = 8

    ys: jax.Array = eqx.field(default=None, static=False)
    zs: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        if self.key is None:
            self.ys = jnp.ones((self.trial_dim, self.ts.shape[1], self.rank))
            self.zs = jnp.ones((self.rank, self.ts.shape[1], self.dim))
        else:
            key, subkey = random.split(self.key)
            self.ys = random.normal(subkey, (self.trial_dim, self.ts.shape[1], self.rank))
            key, subkey = random.split(key)
            self.zs = random.normal(subkey, (self.rank, self.ts.shape[1], self.dim))/jnp.sqrt(self.rank)

    def get_ys(self):

        return jnp.einsum('ijk,kjl->ijl', self.ys, self.zs)


class LinearControl(AbstractPath):

    interpolator: Interpolator = eqx.field(static=False)

    t0: float = None
    t1: float = None

    def __init__(self, ts, interpolator_class, **kwargs):

        self.t0, self.t1 = ts[:, 0], ts[:, -1]

        self.interpolator = interpolator_class(ts, **kwargs)

    def evaluate(self, t0, t1=None, left=True):
        del left

        temp = self.interpolator.evaluate(t0)
        #temp = jnp.abs(temp)
        #temp = temp/jnp.sum(temp, axis=-1)
        temp = temp #/ (jnp.linalg.norm(temp, axis=-1)[..., jnp.newaxis] + jnp.finfo(temp).eps)

        return temp if t1 is None else temp*(t1 - t0)

    def __call__(self, ts):

        return self.evaluate(ts)


class PieceWiseConstantControl(AbstractPath):

    ts: jax.Array = eqx.field(static=False)
    dim: int
    key: random.PRNGKey = None
    init_coef: float = eqx.field(default=1.0, static=True)

    t0: float = None
    t1: float = None

    ys: jax.Array = eqx.field(default=None, static=False)

    def __init__(self, ts, dim, key=None, init_coef=1.0):

        self.ts = ts
        self.dim = dim
        self.key = key
        self.init_coef = init_coef

        self.t0, self.t1 = ts[0], ts[-1]

        if self.key is None:
            self.ys = jnp.ones((len(self.ts), self.dim))*self.init_coef
        else:
            self.ys = random.normal(self.key, (len(self.ts), self.dim))*self.init_coef

    def evaluate(self, t0, t1=None, left=True):
        del left

        temp = self.ys[jnp.argmax(t0 < self.ts)-1]

        return temp if t1 is None else temp * (t1 - t0)

    def __call__(self, ts):
        return self.evaluate(ts)

