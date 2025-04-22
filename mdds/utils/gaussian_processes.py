import jax.numpy as jnp
from jax import random


def rational_quadratic(x, s=1.0, alpha=1.0, l=1.0):

    return s*(1+x**2/(2*alpha*l**2))**-alpha


def generate_covariance(ts, kernel):

    C = kernel(ts[:,jnp.newaxis] - ts[jnp.newaxis])

    return C


def gaussian_process(key, time_dim, neuron_dim, trial_dim=1, kernel=rational_quadratic, cholesky_epsilon=10**-5, t_max=10, **kwargs):

    ts = jnp.linspace(0, t_max, time_dim)

    C = generate_covariance(ts, kernel)

    L = jnp.linalg.cholesky(C+cholesky_epsilon*jnp.eye(time_dim))

    latent_gaussians = random.normal(key=key, shape=(trial_dim, time_dim, neuron_dim))

    gp = jnp.einsum('ti,kin->ktn', L, latent_gaussians)

    return ts, gp, C, L


if __name__ == '__main__':

    key = random.PRNGKey(0)

    ts, gp, C, L = gaussian_process(key, 101, 5)

    import matplotlib.pyplot as plt

    for i in range(5):
        plt.plot(ts, gp[i])

    plt.show()
