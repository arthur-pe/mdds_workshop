import quax

import jax.numpy as jnp
import jax.random as random
import jax


class OrthogonalMatrix(quax.ArrayValue):

    in_dim: int
    out_dim: int
    dim: int
    householder: bool = None

    A: jax.Array

    def __init__(self, in_dim, out_dim, key=None):

        self.householder = self.householder if self.householder is not None else True if in_dim != out_dim else False

        self.in_dim, self.out_dim = in_dim, out_dim
        self.dim = max(in_dim, out_dim)

        if key is None:
            self.A = jnp.zeros(shape=(self.dim if not self.householder else min(in_dim, out_dim), self.out_dim))
        else:
            self.A = 5*random.normal(key, shape=(self.dim if not self.householder else min(in_dim, out_dim), self.out_dim))

    def materialise(self):

        if self.householder:

            U = self.householder_product(self.A)

            return U[:self.out_dim, :self.in_dim]

        else:
            return jax.scipy.linalg.expm(self.A - self.A.T)[:self.out_dim, :self.in_dim]

    @staticmethod
    def householder_product(V):

        V = V / (jnp.sum(V**2, axis=1)[..., jnp.newaxis] + jnp.finfo(V).eps)**0.5

        temp = jnp.eye(V.shape[1])

        for i in range(V.shape[0]):

            temp = temp - 2*jnp.outer(jnp.dot(temp, V[i]), V[i])

        return temp

    def aval(self):
        return jax.core.ShapedArray((self.out_dim, self.in_dim), self.A.dtype)


if __name__ == '__main__':

    O = OrthogonalMatrix(3, 3, jax.random.PRNGKey(0)).materialise()

    print((O @ O.T))
