from . import models
from .decoders import build_decoder

from mdds.controls import build_control, build_low_rank_control

from jax import numpy as jnp
from jax import random

import optax

import equinox as eqx
from diffrax import *

import inspect

classes = inspect.getmembers(models, inspect.isclass)
mdds_class_dict = {name: cls for name, cls in classes if cls.__module__ == 'mdds.mdds.models'}


def build_model(t0, t1, trial_dim, neuron_dim, embedding_dim, manifold_dim, model_key, control_ts,
                vector_fields_class, vector_fields_hyperparameters, data=None, decoder_key=None):

    MDDS = mdds_class_dict[vector_fields_class]

    vector_fields = MDDS(embedding_dim, manifold_dim, model_key, **vector_fields_hyperparameters)

    #subkeys = random.split(model_key, trial_dim)
    key, subkey = random.split(model_key)
    controls = build_low_rank_control(control_ts, manifold_dim, trial_dim, 10, subkey)
    #controls = build_control(control_ts, jnp.ones((trial_dim, manifold_dim)), subkey)

    decoder = build_decoder(embedding_dim, neuron_dim, decoder_key, data)

    return {'vector_fields': vector_fields, 'controls': controls, 'decoder': decoder}


def build_solver(rtol=1e-5, atol=1e-6, jump_ts=jnp.array([])):

    solver = Heun()
    stepsize_controller = PIDController(rtol=rtol, atol=atol, jump_ts=jump_ts, dtmax=0.01, dtmin=0.001)

    return solver, stepsize_controller


def build_optimizer(model, learning_rate, weight_decay):

    if weight_decay != 0:
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = optax.adam(learning_rate=learning_rate, nesterov=True)

    #optimizer = optax.chain(optimizer, optax.clip_by_global_norm(1.0))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    return optimizer, opt_state
