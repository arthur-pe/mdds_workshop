import scipy.stats

from mdds.utils.mappings import Isometry, Linear
from mdds.utils.gaussian_processes import gaussian_process

import equinox as eqx

from jax import numpy as jnp
import jax
from jax import random

from functools import partial


def softplus(x, beta=3):

    return jnp.log(1+jnp.exp(beta*x))/beta


def inv_softplus(x, beta=3):

    return jnp.log(jnp.exp(beta*x)-1)/beta


def re_tanh(x):

    return jnp.tanh(jax.nn.relu(x))


class CalciumConvolution(eqx.Module):

    dim: int = None
    time_dim: int = None

    decay_rate: jax.Array = eqx.field(default=None, static=True)
    ts: jax.Array = eqx.field(default=None, static=True)

    def __post_init__(self):

        self.decay_rate = jnp.full(self.dim, 20.0)
        self.ts = jnp.linspace(0, 1, self.time_dim)

    def convolution(self, decay_rates):

        #filters = jnp.exp(-self.ts[:, jnp.newaxis] * decay_rates[jnp.newaxis, :]) + jnp.exp(-(self.ts[:, jnp.newaxis]) * decay_rates[jnp.newaxis, :])

        filters_1 = (1 - jnp.exp((-self.ts[:, jnp.newaxis]) * decay_rates[jnp.newaxis, :])) #* decay_rates
        filters_2 = jnp.exp(-self.ts[:, jnp.newaxis] * decay_rates[jnp.newaxis, :])
        filters = jnp.concatenate([filters_1[:5], filters_2*filters_1[5]])
        #filters = jnp.exp(-(0.5 - self.ts[:, jnp.newaxis])**2*self.decay_rate[jnp.newaxis, :]*200)

        return filters

    def __call__(self, x, *args, **kwargs):

        x_convolved = jax.vmap(partial(jnp.convolve, mode='full'), in_axes=(-1, -1), out_axes=-1)(x, self.convolution(self.decay_rate))

        return x_convolved[..., :x.shape[-2], :]


class ZScore(eqx.Module):

    s: jax.Array = eqx.field(default=None, static=True)
    b: jax.Array = eqx.field(default=None, static=True)

    def __init__(self, data):

        flat_data = data.reshape(-1, data.shape[-1])

        self.s = jnp.std(flat_data, axis=0)
        self.b = jnp.mean(flat_data, axis=0)

    def __call__(self, x, neuron_ids):

        return x[neuron_ids] * self.s[neuron_ids] + self.b[neuron_ids]


class SVD(eqx.Module):

    '''S: jax.Array = eqx.field(default=None, static=True)
    V: jax.Array = eqx.field(default=None, static=True)
    b: jax.Array = eqx.field(default=None, static=True)'''
    A: dict = eqx.field(default=None)

    def __init__(self, data):

        data = data[~jnp.isnan(data)[..., 0]]

        flat_data = data.reshape(-1, data.shape[-1])

        b = jnp.mean(flat_data, axis=0)

        Q, S, V = jnp.linalg.svd(flat_data - b, full_matrices=False)

        S = S/jnp.sqrt(flat_data.shape[0])

        self.A = {'S': S, 'V': V, 'b': b}

    def __call__(self, x, neuron_ids=slice(None)):
        A = jax.lax.stop_gradient(self.A)

        S, V, b = A['S'], A['V'], A['b']
        S, V, b = S, V[:, neuron_ids], b[neuron_ids]

        if x.shape[-1] > S.shape[0]:
            S = jnp.concatenate([S, jnp.ones((x.shape[-1] - S.shape[0]))])
            V = jnp.concatenate([V, jnp.ones((x.shape[-1] - V.shape[0], V.shape[-1]))], axis=0)
        else:
            S = S[:x.shape[-1]]
            V = V[:x.shape[-1]]
        return (x @ (S[:, jnp.newaxis] * V)) + b


class Decoder(eqx.Module):

    decoders: list[eqx.Module] = eqx.field(default=None, static=False)

    def __init__(self, decoders):

        self.decoders = decoders

    def __call__(self, x, *args):

        #return x[..., args[0]]

        for d in self.decoders:
            x = d(x, *args)

        return x

class Activation(eqx.Module):

    activation: jax.nn.softplus

    def __call__(self, x, *args):

        return self.activation(x)


def build_decoder(embedding_dim, neuron_dim, decoder_key, data):

    decoder = Isometry(embedding_dim, neuron_dim, key=decoder_key, bias=True)
    #decoder = Linear(embedding_dim, neuron_dim, key=random.PRNGKey(0))

    if data is not None:

        data_type = 'rates' if jnp.issubdtype(data.dtype, jnp.floating) else 'spiking'
        #data_type = 'calcium'

        decoder = eqx.tree_at(lambda decoder_: decoder_.b, decoder, jnp.zeros(neuron_dim))

        #z_score = ZScore(data)
        svd = SVD(data)

        match data_type:
            case 'spiking':
                decoder = Decoder([decoder, svd, Activation(softplus)])

            case 'rates':
                decoder = Decoder([decoder, svd])
                #decoder = Decoder([lambda x, args: x])

            #case 'calcium': # Not working
            #    decoder = Decoder([decoder, Activation(softplus), z_score, CalciumConvolution(neuron_dim, data.shape[1])])

    # We assume that there is always a time dimension (useful for time convolution)
    decoder = eqx.filter_vmap(decoder, in_axes=(0, None), out_axes=0)

    return decoder


if __name__ == '__main__':

    from jax import random
    import numpy as np
    from matplotlib import pyplot as plt

    key = random.PRNGKey(0)

    neuron_dim = 5
    time_dim = 100

    data = np.zeros((time_dim, neuron_dim))

    #data[30, :] = 1.0
    data[40, :] = 1.0
    #data[42, :] = 1.0
    #data = np.stack([jnp.sin(jnp.linspace(0, 10, 100)) for _ in range(10)], axis=-1)

    subkey, key = random.split(key)
    ts, data, _, _ = gaussian_process(subkey, time_dim, neuron_dim, trial_dim=2)
    data = jax.nn.relu(data/10)
    data = scipy.stats.poisson.rvs(data)
    #data = np.stack([data]*3, axis=1).reshape([data.shape[0]*3]+[data.shape[1]])[:data.shape[0]//3]
    #data = scipy.ndimage.gaussian_filter1d(data.astype(float), sigma=1, axis=0)

    subkey, key = random.split(key)
    decoder = build_decoder(neuron_dim, neuron_dim, subkey, random.normal(key, data.shape))

    def decoder_vmap(decoder, xs, neuron_ids=slice(None)):

        # return jax.vmap(jax.vmap(decoder, in_axes=(0, None), out_axes=0), in_axes=(0, None), out_axes=0)(xs, neuron_ids)
        return jax.vmap(decoder, in_axes=(0, None), out_axes=0)(xs - xs.reshape(-1, xs.shape[-1]).mean(axis=0), neuron_ids)

    decoded_data = decoder_vmap(decoder, data)

    print(data.shape, decoded_data.shape)

    plt.figure(figsize=(5, neuron_dim), constrained_layout=True)

    d_n = max(np.max(data), np.max(decoded_data))*1.5
    for i in range(neuron_dim):
        plt.plot(data[0, :, i] + i*d_n, color='r')
        plt.plot(decoded_data[0, :, i] + i*d_n, color='b', linestyle='--')

    plt.yticks()

    plt.show()

