import jax.random as random
import jax
import equinox as eqx
import jax.numpy as jnp

import quax

from .orthogonal import OrthogonalMatrix


class SimultaneouslyDiagonalizableMatrices(quax.ArrayValue):

    dim: int
    number_matrices: int
    Ls: jax.Array
    Ss: jax.Array
    U: OrthogonalMatrix

    def __init__(self, dim, number_matrices, key_diag=None, key_off_diag=None, key_orthogonal=None):

        self.dim = dim
        self.number_matrices = number_matrices

        if key_off_diag is not None:
            self.Ls = random.uniform(key_off_diag, (self.number_matrices, self.dim//2))/jnp.sqrt(self.dim)
        else:
            self.Ls = jnp.ones((self.number_matrices, self.dim//2))/jnp.sqrt(self.dim)

        if key_diag is not None:
            self.Ss = random.uniform(key_diag, (self.number_matrices, -(-self.dim//2)))/jnp.sqrt(self.dim)
        else:
            self.Ss = jnp.ones((self.number_matrices, -(-self.dim//2)))/jnp.sqrt(self.dim)

        self.U = OrthogonalMatrix(self.dim, self.dim, key_orthogonal)

    @staticmethod
    @eqx.filter_vmap(in_axes=(None, 0), out_axes=0)
    def matmul_vmap(U, Y):
        return U @ Y @ U.T

    @staticmethod
    @eqx.filter_vmap(in_axes=(0, None), out_axes=0)
    def diag_vmap(S, k):
        return jnp.diag(S, k)

    @staticmethod
    @eqx.filter_vmap(in_axes=(0,0), out_axes=0)
    def interleave(A, B):
        return jnp.stack([A, B], axis=0).reshape(-1, order='F')

    def materialise(self):

        U = self.U.materialise()
        upper_diag = self.interleave(self.Ls, jnp.zeros_like(self.Ls))
        Y = self.diag_vmap(upper_diag[:, :self.dim-1], 1) - self.diag_vmap(upper_diag[:, :self.dim-1], -1)

        diag = self.interleave(self.Ss, self.Ss)
        S = self.diag_vmap(diag[:,:self.dim], 0)

        W = self.matmul_vmap(U, Y+S)

        return W

    def aval(self):
        return jax.core.ShapedArray((self.number_matrices, self.dim, self.dim), self.Ls.dtype)
