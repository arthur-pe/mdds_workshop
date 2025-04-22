import quax

from mdds.parameterization import OrthogonalMatrix

import jax.random as random
import jax
import jax.numpy as jnp

import equinox as eqx
import quax

from functools import partial


class Isometry(eqx.Module):

    in_dim: int
    out_dim: int
    dim: int
    bias: bool
    scaling: bool

    b: jax.Array
    s: jax.Array
    c: float = eqx.field(default=0.0, static=False, converter=jax.numpy.asarray)
    U: OrthogonalMatrix

    def __init__(self, in_dim, out_dim, key=None, bias=True, scaling=False):

        self.in_dim, self.out_dim = in_dim, out_dim
        self.dim = max(in_dim, out_dim)
        self.bias = bias
        self.scaling = scaling

        if key is None:
            self.b = jnp.zeros(shape=(self.out_dim,))
            self.s = jnp.ones((self.out_dim,))

            self.U = jnp.zeros(shape=(self.dim, self.dim))

        else:
            key, subkey = random.split(key)
            self.b = random.normal(subkey, shape=(self.out_dim,))/3

            key, subkey = random.split(key)
            self.s = random.normal(subkey, shape=(self.out_dim,)) / 3

            key, subkey = random.split(key)
            self.U = random.normal(subkey, shape=(self.dim, self.dim)) / jnp.sqrt(self.dim)

        #self.U = OrthogonalMatrix(in_dim, out_dim, key)

    '''@quax.quaxify
    def __call__(self, x, neuron_ids, key=None):

        temp = (self.c+1)*(self.U[neuron_ids] @ x)

        if self.scaling:
            temp = temp * self.s[neuron_ids]

        if self.bias:
            temp = temp + self.b[neuron_ids]

        return temp'''

    #@quax.quaxify
    def __call__(self, x, neuron_ids=slice(None), key=None):

        #temp = (self.c+1)*(self.U @ x)
        U = jax.scipy.linalg.expm(self.U - self.U.T)
        temp = U[:self.out_dim, :self.in_dim] @ x

        if self.scaling:
            temp = temp * (self.s + 1)

        if self.bias:
            temp = temp + self.b

        return temp


class Linear(eqx.Module):

    in_dim: int
    out_dim: int
    dim: int
    bias: bool

    b: jax.Array
    M: jax.Array

    def __init__(self, in_dim, out_dim, key=None, bias=False):

        self.in_dim, self.out_dim = in_dim, out_dim
        self.dim = max(in_dim, out_dim)
        self.bias = bias

        if key is None:
            self.b = jnp.zeros(shape=(self.out_dim,))

        else:
            key, subkey = random.split(key)
            self.b = random.normal(subkey, shape=(self.out_dim,))/3

        key, subkey = random.split(key)
        self.M = random.normal(subkey, shape=(self.out_dim, self.in_dim))/jnp.sqrt(in_dim)

    def __call__(self, x, neuron_ids=slice(None), key=None):

        temp = self.M @ x

        if self.bias:
            temp = temp + self.b

        return temp

