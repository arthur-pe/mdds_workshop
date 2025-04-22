from diffrax import SubSaveAt, SaveAt, Solution

from ..utils import decoder_vmap
from ..evaluate import evaluate

from mdds.controls import resolve_parameterization,  batch_controls, evaluate_control

import jax
from jax import numpy as jnp
from jax import random

import equinox as eqx


def softplus(x, beta=1): return jnp.log(1+jnp.exp(beta*x))/beta


def poisson_ll(rate_estimates, data_spikes, mask):

    rate_estimates = rate_estimates + jnp.finfo(rate_estimates).eps

    return jnp.sum(-jnp.log(jnp.power(rate_estimates, data_spikes)*jnp.exp(-rate_estimates)/jax.scipy.special.factorial(data_spikes)))/(jnp.sum(mask.astype(float))*jnp.sqrt(rate_estimates.shape[-1]))


def l_p(x, y, mask, p=2):

    norms = jnp.sum(jnp.abs(x-y)**p, axis=-1)#**(1/p)
    '''norms = (1-jnp.cos(jnp.sum(x*y, axis=-1)/(jnp.linalg.norm(x, axis=-1)*jnp.linalg.norm(y, axis=-1) + 10**-6)))/2#**(1/p)
    return jnp.sum(norms)/jnp.sum(mask.astype(float))'''
    return jnp.sum(norms)/(jnp.sum(mask.astype(float))*jnp.sqrt(x.shape[-1]))


def safe_divide(x, y, eps=10**-6):

    f = lambda y: (-eps <= y) & (y <= eps)

    y_ = jnp.where(f(y), 0, y)
    return jnp.where(f(y), 0, x / y_)


def sparsity(x, axis=-1, order=2):

    l1_l2 = safe_divide(jnp.linalg.norm(x, axis=axis, ord=order), jnp.linalg.norm(x, axis=axis, ord=2*order))
    #l1_l2 = jnp.linalg.norm(x, axis=axis, ord=order) / jnp.linalg.norm(x, axis=axis, ord=2 * order)

    sqrt_k = (x.shape[axis]) ** (1 / (2 * order))

    return (l1_l2 - 1) / (sqrt_k - 1)


def mean_var(x, y, mask):

    sum_mask = jnp.sum(mask.astype(float))

    y_mean = jnp.sum(y.reshape(-1, y.shape[-1]), axis=0)/sum_mask
    y_mean = mask[..., jnp.newaxis]*y_mean[jnp.newaxis, jnp.newaxis]

    # sum(||y-x||^2/||y-y_hat||^2)
    norms = jnp.sum((y-x)**2, axis=(0, 1, 2))/jnp.sum((y-y_mean)**2, axis=(0, 1, 2))

    return norms

    #return (x - y)/sum_mask


@eqx.filter_jit
def losses(model, stepsize_controller, saveat, solver, t0, intrinsic_noise, extrinsic_noise, data, trial_batch_ids, neuron_batch_ids,
         time_mask, frobenius_penalty, key):
    vector_fields, controls, decoder = model['vector_fields'], model['controls'], model['decoder']

    vector_fields = resolve_parameterization(vector_fields)

    controls = batch_controls(controls, trial_batch_ids)

    #controls_parameterized = jax.vmap(resolve_parameterization, in_axes=(0,), out_axes=0)(controls)
    controls_parameterized = resolve_parameterization(controls)

    # ===== Batching =====
    time_mask = time_mask[trial_batch_ids]
    saveat = saveat[trial_batch_ids]
    data = data[trial_batch_ids][..., neuron_batch_ids]

    # ===== Mask saveat and set nans to t0 or tmax =====
    saveat = jnp.where(time_mask, saveat,  jnp.nan)
    saveat_argmin = jnp.nanmin(saveat, axis=1)

    def f(carry, inputs):
        temp = jnp.where(jnp.isnan(inputs), carry, inputs)
        return temp, temp
    saveat = jax.lax.scan(f, saveat_argmin, saveat.T)[1].T

    # ===== Evaluate =====

    subkeys = random.split(key, trial_batch_ids.shape[0])

    t1 = jnp.max(saveat, axis=1)

    preparatory_saveat = jnp.tile(jnp.linspace(t0, 0, saveat.shape[1])[:-1], (saveat.shape[0], 1))
    saveat_obj = SaveAt(subs={'data':SubSaveAt(ts=saveat)} | ({} if frobenius_penalty == 0 else {'preparatory':SubSaveAt(ts=preparatory_saveat)}))

    solution = evaluate(subkeys, vector_fields, controls_parameterized, stepsize_controller, saveat_obj, solver, t0, t1, intrinsic_noise, extrinsic_noise)
    ys = solution.ys['data'] #+ model['vector_fields'].b

    #solution_preparatory = evaluate(subkeys, vector_fields, controls_parameterized, stepsize_controller, preparatory_saveat, solver, t0, t1, intrinsic_noise, extrinsic_noise)

    # ===== Loss =====
    loss_fn = l_p if jnp.issubdtype(data, jnp.floating) else poisson_ll

    data_masked = jnp.where(time_mask[..., jnp.newaxis], data, 0)
    ys_decoded = decoder_vmap(decoder, ys, neuron_batch_ids)
    ys_decoded_masked = jnp.where(time_mask[..., jnp.newaxis], ys_decoded, 0)

    l_data = loss_fn(ys_decoded_masked, data_masked, time_mask)

    time_mask_ = jnp.concatenate([jnp.ones(time_mask.shape, dtype=jnp.bool)[:,:-1], time_mask], axis=1)
    if frobenius_penalty != 0:

        zs = jnp.concatenate(list(solution.ys.values()), axis=1)
        s = jnp.mean(jnp.sum(jax.vmap(jax.vmap(vector_fields.F))(zs) ** 2, axis=-2), axis=-1)[..., jnp.newaxis]  # jnp.std(zs, axis=(0, 1))
        x = random.normal(key, zs.shape) * 3#* 0.3  # *s*0.3 #+ model['vector_fields'].b
        # x = random.normal(key, ys.shape)*3#*jnp.nanstd(data)*10 # Times var of data?
        # x = x + vector_fields.b

        l_lie_bracket = eqx.filter_vmap(eqx.filter_vmap(vector_fields.regularization))(x)#.mean()
        l_lie_bracket = jnp.where(time_mask_, l_lie_bracket, 0).sum()/time_mask_.astype(float).sum()
    else:
        l_lie_bracket = 0

    '''x = random.normal(key, (10,) + ys.shape)*0.1*jnp.std(ys)
    l_lie_bracket = jnp.where(time_mask[jnp.newaxis],
                              eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(vector_fields.regularization, in_axes=(0,)), in_axes=(0,)), in_axes=(0,))(ys[jnp.newaxis] + x),
                              0).sum()/jnp.sum(time_mask)'''

    '''x = random.normal(key, ys.shape).reshape(-1, ys.shape[-1])#*jnp.nanstd(data)*10 # Times var of data?
    x = x/jnp.linalg.norm(x, axis=-1)[..., jnp.newaxis]
    key, subkey = random.split(key)
    x = x + random.normal(subkey, ys.shape).reshape(-1, ys.shape[-1])*0.1
    l_lie_bracket = eqx.filter_vmap(vector_fields.regularization, in_axes=(0,))(x).mean() if frobenius_penalty != 0 else 0'''

    preparatory_saveat = jnp.concatenate([jnp.tile(jnp.linspace(t0, 0, saveat.shape[1])[:-1], (saveat.shape[0], 1)), saveat], axis=1)
    evaluated_controls = evaluate_control(controls_parameterized, preparatory_saveat)
    # evaluated_controls = (evaluated_controls[:, 1:] - evaluated_controls[:, :-1])/(saveat.subs.ts[1:] - saveat.subs.ts[:-1])[:, jnp.newaxis]
    p = 1.0
    #l_controls = l_p_(evaluated_controls, 0, time_mask, p=2) + l_p_(evaluated_controls, 0, time_mask, p=1)
    l_controls = jnp.abs(evaluated_controls).mean()
    #evaluated_controls = evaluate_control(model['controls'], saveat.subs.ts)[trial_batch_ids]
    # l_controls = (jnp.abs(evaluated_controls).mean(axis=(0, 2))**2).mean()
    # l_controls = jnp.abs(evaluated_controls).mean()

    #l_vf = jnp.mean(jnp.abs(jax.vmap(jax.vmap(vector_fields.F_norm))(x)))

    l_vf = jnp.abs(jax.vmap(vector_fields.F)(ys.reshape(-1, ys.shape[-1]))).mean()
    l = (l_data+ l_controls*0.01  + l_vf*0.01, l_lie_bracket)  # + l_lie_bracket*0.1 + l_lie_bracket*0.001 + frobenius_penalty*0.01 + l_vf*0.01

    mean_variance = mean_var(ys_decoded_masked, data_masked, time_mask)

    solution = Solution(t0=solution.t0, t1=solution.t1, interpolation=solution.interpolation, stats=solution.stats,
                        result=solution.result, solver_state=solution.solver_state, controller_state=solution.controller_state,
                        made_jump=solution.made_jump, event_mask=solution.event_mask,
                        ts=solution.ts['data'], ys=solution.ys['data'])

    return l, (solution, mean_variance, l_lie_bracket) # l_lie_bracket + jnp.round(l_controls*1000,0)


def loss(*args, **kwargs):

    ls, aux = losses(*args, **kwargs)

    return ls[0], aux

def loss_(*args, **kwargs):

    ls, aux = losses(*args, **kwargs)

    return ls[0]#, aux


from mdds.mdds.projected_grad.grad import value_projected_grad_aux, value_grad_aux


@eqx.filter_jit
@value_projected_grad_aux
def loss_grad(model_diff, model_static, *args):

    model = model_diff | model_static

    return losses(model, *args)
