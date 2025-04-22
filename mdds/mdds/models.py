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

        Js = jax.jacrev(self.F_norm)(x)
        Fs = self.F_norm(x)

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

        Js = jax.jacrev(self.F_norm)(x)
        Fs = self.F_norm(x)

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

        #q = q.at[jnp.diag_indices(q.shape[0])].add(jnp.finfo(basis).eps)

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


class LinearOrthogonalMDDS(BaseMDDS):

    Ws: SimultaneouslyDiagonalizableMatrices = eqx.field(default=None, static=False)
    b: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        key, *subkeys = random.split(key, 4)
        self.Ws = SimultaneouslyDiagonalizableMatrices(self.dim, self.intrinsic_dim, *subkeys)

    @staticmethod
    def householder_product(V):
        """
        :param V: m x n
        :return: m x n
        """

        V = V / (jnp.sum(V**2, axis=1)[..., jnp.newaxis] + jnp.finfo(V).eps)**0.5

        temp = jnp.eye(V.shape[0], V.shape[1])

        def f(carry, x): return carry - 2*jnp.outer(jnp.dot(carry, x), x), None

        return jax.lax.scan(f, temp, V)[0]

    @staticmethod
    @eqx.filter_vmap(in_axes=(0, None), out_axes=1)
    def matmul_vmap(W, x):
        return W @ x

    @quax.quaxify
    def F(self, x):

        temp = self.matmul_vmap(self.Ws, x + self.b)

        return temp


class OrthogonalMDDS(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh
    mlp: eqx.nn.MLP = eqx.field(default=None, static=False)
    b: jax.Array = eqx.field(default=None, static=False)
    mlp2: eqx.nn.MLP = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.mlp = eqx.nn.MLP(self.dim, self.dim*self.intrinsic_dim,
                              self.mlp_width, self.mlp_depth, activation=self.activation, use_final_bias=False, key=subkey)

        key, subkey = random.split(key)
        self.b = random.normal(subkey, (self.dim,))/3
        '''key, subkey = random.split(key)
        self.mlp2 = eqx.nn.MLP(self.dim, self.intrinsic_dim,
                              self.mlp_width, self.mlp_depth, activation=self.activation, key=subkey)'''

    '''@staticmethod
    def householder_product_(V):

        V = V / (jnp.sum(V**2, axis=1)[..., jnp.newaxis] + jnp.finfo(V).eps)**0.5

        temp = jnp.eye(V.shape[1])

        for i in range(V.shape[0]):

            temp = temp - 2*jnp.outer(jnp.dot(temp, V[i]), V[i])

        return temp'''

    @staticmethod
    def householder_product(V):

        V = V / (jnp.sum(V**2, axis=1)[..., jnp.newaxis] + jnp.finfo(V).eps)**0.5

        temp = jnp.eye(V.shape[0], V.shape[1])

        def f(carry, x): return carry - 2*jnp.outer(jnp.dot(carry, x), x), None

        return jax.lax.scan(f, temp, V)[0]

    def F(self, x):

        A = self.mlp(x+self.b).reshape(self.intrinsic_dim, self.dim)

        U = self.householder_product(A)
        #I = jnp.eye(self.dim)
        #B = A - A.T
        #U = jnp.matmul((I - B), jnp.linalg.inv(I + B))
        #U = I - B @ B.T
        # We can compute an orthogonal matrix either through the matrix exponential or the householder product
        #U = jax.scipy.linalg.expm((A - A.T)*0.1)

        return U.T#*(1+self.mlp2(x)/10)


class TensorMDDS(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh
    b: jax.Array = eqx.field(default=None, static=False)
    bs: list[jax.Array] = eqx.field(default=None, static=False)
    W_neuron: list[jax.Array] = eqx.field(default=None, static=False)
    W_vf: list[jax.Array] = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))#/jnp.sqrt(self.dim)

        key, *subkeys = random.split(key, self.mlp_depth+2)
        self.bs = [random.normal(k, shape=(self.dim, self.intrinsic_dim))/jnp.sqrt(self.dim) for k in subkeys[:-1]]

        key, *subkeys = random.split(key, self.mlp_depth+2)
        self.W_neuron = [random.normal(k, shape=(self.intrinsic_dim, self.dim, self.dim))/jnp.sqrt(self.dim) for k in subkeys]

        key, *subkeys = random.split(key, self.mlp_depth+2)
        self.W_vf = [random.normal(k, shape=(self.intrinsic_dim, self.intrinsic_dim))/jnp.sqrt(self.intrinsic_dim) for k in subkeys]
        #self.W_vf = [jnp.eye(self.intrinsic_dim) for k in subkeys]

        '''key, subkey = random.split(key)
        self.mlp2 = eqx.nn.MLP(self.dim, self.intrinsic_dim,
                              self.mlp_width, self.mlp_depth, activation=self.activation, key=subkey)'''

    def tensor_mlp(self, x):

        x = x + self.b

        x = jnp.tile(x, (self.intrinsic_dim, 1)).T

        for W_n, W_v, b in zip(self.W_neuron, self.W_vf, self.bs+[0]):

            x = self.activation(x)
            x = jnp.einsum('kij,jk->ik', W_n, x) #- W_n.transpose(0, 2, 1)
            x = self.activation(x)
            x = jnp.einsum('ij,kj->ki', W_v, x)+b

        return x

    def F(self, x):

        A = self.tensor_mlp(x)

        return A


class AdaptiveMDDS(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh
    mlp: eqx.nn.MLP = eqx.field(default=None, static=False)
    mlp2: eqx.nn.MLP = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.mlp = eqx.nn.MLP(self.dim, self.dim*self.intrinsic_dim,
                              self.mlp_width, self.mlp_depth, activation=self.activation, use_final_bias=False, key=subkey)

    def F(self, x):

        U = self.mlp(x).reshape(self.dim, self.intrinsic_dim)

        return U #/ (jnp.linalg.norm(U, axis=0) + jnp.finfo(U).eps)

    @staticmethod
    def sparsity(x, axis=-1):

        l1_l2 = safe_divide(jnp.linalg.norm(x, axis=axis, ord=4), jnp.linalg.norm(x, axis=axis, ord=2))
        sqrt_k = (x.shape[axis])**(1/4)

        return (sqrt_k - l1_l2)/(sqrt_k - 1)

    '''def sparsity_penalty(self, x):

        regularization = 0

        if self.lie_bracket_regularization:

            Fs = self.F_norm(x)
            LBs = self.lie_bracket_tensor(x).transpose(2, 1, 0)
            LBs = LBs - self.vectors_on_basis(LBs, Fs.T)
            LBs = LBs.transpose(2, 1, 0)

            regularization = regularization + (LBs**2).mean(axis=0).sum()/(self.intrinsic_dim*(self.intrinsic_dim-1)/2)

        if self.torsion_regularization:
            regularization = regularization + (self.torsion_tensor(x)**2).mean(axis=0).sum()/(self.intrinsic_dim ** 3 - self.intrinsic_dim)

        

        #jax.debug.print('{x}, {y}', x=s, y=jnp.abs(s)/jnp.sum(jnp.abs(s)))
        s = jnp.linalg.norm(self.mlp(x).reshape(self.intrinsic_dim, self.dim), axis=0)
        return self.sparsity(s)'''


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


'''from .sparse_mlp import SparseMLP
class SparseDNNMDDS(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh

    b: jax.Array = eqx.field(default=None, static=False)
    mlp: SparseMLP = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        key, subkey = random.split(key)
        self.mlp = SparseMLP(self.dim, [self.mlp_width]*self.mlp_depth, self.dim*self.intrinsic_dim, 0.05, activation=self.activation, key=subkey)

    def F(self, x):

        Fs = self.mlp(x+self.b).reshape(self.dim, self.intrinsic_dim) #+ self.bs_out

        return Fs#-0.1*x[:, jnp.newaxis]#/(jnp.linalg.norm(Fs, axis=0) + jnp.finfo(Fs).eps)
'''


# Doesn't work
class LinearVariantMDDS(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3

    activation: callable = jnp.tanh
    U: OrthogonalMatrix = eqx.field(default=None, static=False)
    mlp: eqx.nn.MLP = eqx.field(default=None, static=False)
    bs_out: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.U = OrthogonalMatrix(self.dim, self.dim, subkey)

        subkeys = random.split(key, self.intrinsic_dim)
        self.mlp = self.make_ensemble(subkeys, self.dim, self.activation, self.mlp_width, self.mlp_depth)

    @staticmethod
    @eqx.filter_vmap(in_axes=(0, None, None, None, None))
    def make_ensemble(key, dim, activation, width, depth):
        return eqx.nn.MLP(dim, dim, width, depth, activation=activation, key=key)

    @staticmethod
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
    def evaluate_per_ensemble(mlp, x):
        return mlp(x)

    @staticmethod
    @eqx.filter_vmap(in_axes=(0, None))
    def diag_vmap(S, k):
        return jnp.diag(S, k)

    @staticmethod
    @eqx.filter_vmap(in_axes=(None, 0))
    def matmul_vmap(U, Y):
        return U @ Y @ U.T

    @quax.quaxify
    def F(self, x):

        Ss = self.evaluate_per_ensemble(self.mlp, x)
        Ws = self.matmul_vmap(self.U, self.diag_vmap(Ss, 0))

        temp = self.matmul_vmap(Ws, x)

        return temp


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


from typing import Tuple, List, Callable, Optional
class MLP(eqx.Module):
    weights: List[jnp.ndarray]
    biases: List[jnp.ndarray]
    activation: Callable
    activation_inverse: Callable

    def __init__(self, in_features: int, hidden_features: List[int], out_features: int,
                 activation: str = "tanh", *, key):
        """
        Initialize a simple Multi-Layer Perceptron using matrices directly.

        Args:
            in_features: Number of input features
            hidden_features: List of hidden layer sizes
            out_features: Number of output features
            activation: Activation function name ("tanh" or "elu")
            key: JAX random key for parameter initialization
        """
        super().__init__()

        # Set activation function and its inverse
        if activation == "tanh":
            self.activation = jax.nn.tanh
            self.activation_inverse = lambda x: 0.5 * jnp.log((1 + x) / (1 - x))  # arctanh
        elif activation == "elu":
            alpha = 1.0  # Default alpha value for ELU
            self.activation = lambda x: jax.nn.elu(x, alpha)
            # Inverse of ELU
            self.activation_inverse = lambda x: jnp.where(
                x > 0,
                x,  # If x > 0, inverse is x itself
                jnp.log(x / alpha + 1)  # If x ≤ 0, inverse is log(x/alpha + 1)
            )

        # Create sub-keys for each layer
        keys = jax.random.split(key, len(hidden_features) + 1)

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        layer_dims = [in_features] + hidden_features + [out_features]

        # Initialize all layers
        for i in range(len(layer_dims) - 1):
            # Xavier/Glorot initialization for weights
            w = jax.random.normal(keys[i], (layer_dims[i + 1], layer_dims[i])) / jnp.sqrt(layer_dims[i])
            b = jnp.zeros((layer_dims[i + 1],))

            self.weights.append(w)
            self.biases.append(b)

    def f(self, x):
        """Forward pass through the network."""
        for i in range(len(self.weights) - 1):
            x = jnp.dot(self.weights[i], x) + self.biases[i]
            x = self.activation(x)

        # Output layer (without activation)
        return jnp.dot(self.weights[-1], x)# + self.biases[-1]

    def F(self, y):
        """
        Compute the inverse of the network.

        Args:
            y: Output to inverse-map back to input space

        Returns:
            Approximate input x such that f(x) ≈ y
        """
        # Start with the output layer (no activation)
        x = y #- self.biases[-1]

        # Compute pseudoinverse of the last weight matrix
        w_pinv = jnp.linalg.pinv(self.weights[-1])
        x = jnp.dot(w_pinv, x)

        # Go through the layers in reverse order
        for i in range(len(self.weights) - 2, -1, -1):
            # Undo activation
            x = self.activation_inverse(x)

            # Undo linear transformation
            x = x - self.biases[i]
            w_pinv = jnp.linalg.pinv(self.weights[i])
            x = jnp.dot(w_pinv, x)

        return x

    # For backward compatibility
    def __call__(self, x):
        """Alias for f method."""
        return self.f(x)



class DNNMDDS_2(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh

    b: jax.Array = eqx.field(default=None, static=False)
    mlp: eqx.nn.MLP = eqx.field(default=None, static=False)
    mlp_f: eqx.nn.MLP = eqx.field(default=None, static=False)
    bs_out: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        key, subkey = random.split(key)
        self.mlp = MLP(self.dim, [self.mlp_width]*self.mlp_depth, self.dim,'elu', key=subkey)

        key, subkey = random.split(key)
        self.mlp_f = MLP(self.intrinsic_dim, [ self.mlp_width]*self.mlp_depth, self.intrinsic_dim**2, 'elu', key=subkey)

        self.bs_out = jnp.zeros((self.dim, self.intrinsic_dim))
        #self.bs_out = jnp.eye(3)

    def F(self, x):

        y = self.mlp.f(x+self.b)

        Fs = self.mlp_f(y).reshape(self.intrinsic_dim, self.intrinsic_dim) #+ self.bs_out

        Fs = eqx.filter_vmap(self.mlp)(Fs)

        return Fs.T#/(jnp.linalg.norm(Fs, axis=0) + jnp.finfo(Fs).eps)##


class MDDSProd(BaseMDDS):

    mlp_width: int = 30
    mlp_depth: int = 3
    activation: callable = jnp.tanh

    b: jax.Array = eqx.field(default=None, static=False)
    mlp: eqx.nn.MLP = eqx.field(default=None, static=False)
    mlps: eqx.nn.MLP = eqx.field(default=None, static=False)
    bs_out: jax.Array = eqx.field(default=None, static=False)

    def __post_init__(self):

        key, subkey = random.split(self.key)
        self.b = random.normal(subkey, shape=(self.dim,))/jnp.sqrt(self.dim)

        key, subkey = random.split(key)
        self.mlp = eqx.nn.MLP(self.dim, self.dim,
                              self.mlp_width, self.mlp_depth, activation=self.activation, use_final_bias=False, key=subkey)

        keys = random.split(key, self.intrinsic_dim+1)
        self.mlps = [eqx.nn.MLP(self.dim//self.intrinsic_dim, self.dim//self.intrinsic_dim, self.mlp_width, self.mlp_depth, activation=self.activation, use_final_bias=False, key=k) for k in keys[:-1]]
        self.mlps.append(eqx.nn.MLP(self.dim%self.intrinsic_dim, self.dim%self.intrinsic_dim, self.mlp_width, self.mlp_depth, activation=self.activation, use_final_bias=False, key=keys[-1]))

        self.bs_out = jnp.zeros((self.dim, self.intrinsic_dim))
        #self.bs_out = jnp.eye(3)

    def F(self, x):

        y = self.mlp(x + self.b)

        Fs = jnp.concatenate([mlp(y[i*self.dim//self.intrinsic_dim:min((i+1)*self.dim//self.intrinsic_dim, self.dim)]) for i, mlp in enumerate(self.mlps)], axis=-1)  # + self.bs_out

        Fs = jnp.stack([Fs*((jnp.arange(i, self.dim+i)%self.intrinsic_dim)==0) for i in range(self.intrinsic_dim)])

        #Fs = eqx.filter_vmap(self.mlp)(Fs)

        return Fs.T  # /(jnp.linalg.norm(Fs, axis=0) + jnp.finfo(Fs).eps)##


if __name__ == '__main__':

    jax.config.update('jax_platforms', 'cpu')
    jax.config.update("jax_default_device", jax.devices('cpu')[0])

    model = MDDSProd(dim=10, intrinsic_dim=2, key=random.key(0), mlp_depth=1, mlp_width=10, torsion_regularization=False)

    x = jnp.array([-1, 2.0]*5)

    print(model.F(x).shape)

    print(model.regularization(x).shape)
