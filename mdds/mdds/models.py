import quax

from mdds.parameterization import OrthogonalMatrix, SimultaneouslyDiagonalizableMatrices

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Iterable, List
from jax import random

import itertools


def safe_divide(x, y):

    y_ = jnp.where(y == 0.0, 0, y)

    return jnp.where(y == 0.0, 0, x / y_)

def gram_schmidt(vectors):

    vectors = vectors.T

    num_vectors = vectors.shape[-1]

    def body_fn(vecs, i):
        # Slice out the vector w.r.t. which we're orthogonalizing the rest.
        u = jnp.nan_to_num(safe_divide(vecs[:,i], jnp.linalg.norm(vecs[:,i])))
        # Find weights by dotting the d x 1 against the d x n.
        weights = u@vecs
        # Project out vector `u` from the trailing vectors.
        masked_weights = jnp.where(jnp.arange(num_vectors) > i, weights, 0.)
        vecs = vecs - jnp.outer(u,masked_weights)
        return vecs, None

    vectors, _ = jax.lax.scan(body_fn, vectors, jnp.arange(num_vectors - 1))
    vec_norm = jnp.linalg.norm(vectors, axis=0, keepdims=True)

    return jnp.nan_to_num(safe_divide(vectors, vec_norm)).T


class BaseMDDS(eqx.Module):
    """
    This class is inherited by most MDDS models.

    :param dim: the dimension of the space in which the manifold is embedded
    :param intrinsic_dim: the dimension of the manifold
    :param key: pseudo-random number generator key

    :param lie_bracket_regularization: whether to add a regularization to the Lie bracket of the vector fields
    :param torsion_regularization: whether to add a regularization to the torsion of the vector fields
    """

    dim: int
    intrinsic_dim: int
    key: random.PRNGKey

    lie_bracket_regularization: bool = True
    torsion_regularization: bool = False

    def F(self, x):
        """
        The vector fields
        :param x: (dim) state
        :return: (dim, intrinsic_dim) the evaluation of the vector fields at x
        """

        raise NotImplementedError('Vector field not implemented')

    def F_norm(self, x):

        Fs = self.F(x)

        return safe_divide(Fs, jnp.linalg.norm(Fs, axis=0))  # + jnp.finfo(Fs).eps 10**-6

    def f(self, t, x, *args):
        """
        :param t: (scalar) ignored
        :param x: (dim) state
        :param args: ignored
        :return: (dim, intrinsic_dim) the evaluation of the vector fields at x
        """

        return self.F(x)

    def torsion_tensor(self, x):
        """
        Returns the torsion tensor evaluated at x.
        It has entries J_i J_j F_k where J_i is the Jacobian of the ith vector field.

        :param x: (dim) state
        :return: (intrinsic_dim x intrinsic_dim x intrinsic_dim x dim) the torsion tensor"""

        Js = jax.jacrev(self.F)(x)
        Fs = self.F(x)

        J_Xs = jax.vmap(jnp.matmul, in_axes=(0, None))(Js, Fs)

        J_J_Xs = jax.vmap(jnp.matmul, in_axes=(0, None))(Js, J_Xs)

        TTs = J_J_Xs - J_J_Xs - J_J_Xs

        return TTs

    def lie_bracket_tensor(self, x):
        """
        Returns the Lie bracket tensor evaluated at x.
        It has entries J_i F_j where J_i is the Jacobian of the ith vector field.

        :param x: (dim) state
        :return: (dim x intrinsic_dim x intrinsic_dim) the Lie bracket tensor"""

        Js = jax.jacrev(self.F)(x)
        Fs = self.F(x)

        J_Xs = jax.vmap(jnp.matmul, in_axes=(0, None))(Js, Fs)

        LBs = J_Xs - J_Xs.transpose(0, 2, 1)

        return LBs

    @staticmethod
    def vectors_on_basis(vectors, basis):
        """
        Projects the vectors on a non-orthogonal basis
        https://math.stackexchange.com/questions/105753/cayley-transform-exponential-mapping-and-more
        :param vectors: ... x n
        :param basis: k x n
        :return: ... x n
        """

        q = basis @ basis.T

        q = q.at[jnp.diag_indices(q.shape[0])].add(jnp.finfo(basis).eps)

        projection = basis.T @ jnp.linalg.inv(q) @ basis

        return vectors @ projection

    def regularization(self, x):
        """
        Returns what will be added to the regularization term of the loss.
        Note that coefficients are in the loss function.

        :param x: (dim) state
        :return: (intrinsic_dim x intrinsic_dim x dim) the Lie bracket tensor"""

        regularization = 0

        if self.lie_bracket_regularization:

            Fs = self.F(x)
            LBs = self.lie_bracket_tensor(x).transpose(2, 1, 0) # 3 x 2 x 2_ -> 2_ x 2 x 3
            LBs = LBs - self.vectors_on_basis(LBs, Fs.T)
            LBs = LBs.transpose(2, 1, 0)

            Fs_norm = (Fs**2).sum(axis=0)
            LBs_norm = (LBs**2).sum(axis=0)

            #regularization = regularization + safe_divide(LBs_norm.sum(), self.intrinsic_dim*(self.intrinsic_dim-1)/2)
            regularization = regularization + safe_divide(LBs_norm, jnp.outer(Fs_norm, Fs_norm)*self.intrinsic_dim*(self.intrinsic_dim-1)/2).sum()

        if self.torsion_regularization:
            regularization = regularization + (self.torsion_tensor(x)**2).mean(axis=0).sum()/(self.intrinsic_dim ** 3 - self.intrinsic_dim)

        return regularization

    def get_initial_state(self):
        """
        :return: (dim) the initial state"""

        return jnp.zeros((self.dim,))


class LinearMDDS(BaseMDDS):

    Ws: SimultaneouslyDiagonalizableMatrices = eqx.field(default=None, static=False)
    b: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        key, *subkeys = random.split(key, 4)
        self.Ws = SimultaneouslyDiagonalizableMatrices(self.dim, self.intrinsic_dim, *subkeys)

    @staticmethod
    @eqx.filter_vmap(in_axes=(0, None), out_axes=1)
    def matmul_vmap(W, x):
        return W @ x

    @quax.quaxify
    def F(self, x):

        temp = self.matmul_vmap(self.Ws, x + self.b)

        return temp


class DNNMDDS(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh

    b: jax.Array = eqx.field(default=None, static=False)
    mlp: eqx.nn.MLP = eqx.field(default=None, static=False)
    bs_out: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        key, subkey = random.split(key)
        self.mlp = eqx.nn.MLP(self.dim, self.dim*self.intrinsic_dim,
                              self.mlp_width, self.mlp_depth, activation=self.activation, use_final_bias=False, key=subkey)

        self.bs_out = jnp.zeros((self.dim, self.intrinsic_dim))
        #self.bs_out = jnp.eye(3)

    def F(self, x):

        Fs = self.mlp(x+self.b).reshape(self.dim, self.intrinsic_dim) + self.bs_out

        return Fs#/(jnp.linalg.norm(Fs, axis=0) + jnp.finfo(Fs).eps)##



class LinearMDDS_(BaseMDDS):

    Ws: jax.Array = eqx.field(default=None, static=False)
    b: jax.Array = eqx.field(default=None, static=False)
    bs_out: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        _, subkey = random.split(key, 2)
        #self.Ws = random.normal(subkey, (self.intrinsic_dim, self.dim, self.dim))/jnp.sqrt(self.dim)
        self.Ws = jax.vmap(jnp.diag)(random.normal(subkey, (self.intrinsic_dim, self.dim)))

        '''key, *subkeys = random.split(key, 4)
        self.Ws = self.Ws.at[:].set(SimultaneouslyDiagonalizableMatrices(self.dim, self.intrinsic_dim, *subkeys).materialise())'''

        self.bs_out = jnp.zeros((self.dim, self.intrinsic_dim))

    @staticmethod
    @eqx.filter_vmap(in_axes=(0, None), out_axes=1)
    def matmul_vmap(W, x):
        return W @ x

    def F(self, x):

        temp = self.matmul_vmap(self.Ws, x + self.b) #+ self.bs_out

        return temp# * jnp.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]).T

