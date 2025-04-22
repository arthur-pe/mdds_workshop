import equinox as eqx
from diffrax import ControlTerm, diffeqsolve, RecursiveCheckpointAdjoint, BacksolveAdjoint, SaveAt, MultiTerm, VirtualBrownianTree
import jax
import lineax

@eqx.filter_vmap(in_axes=(0, None, eqx.if_array(0), None, eqx.if_array(0), None, None, 0, None, None))
def evaluate(key, vector_fields, controls, stepsize_controller, saveat, solver, t0, t1, intrinsic_noise, extrinsic_noise):

    y0 = vector_fields.get_initial_state()

    terms = [ControlTerm(vector_fields.f, controls)]

    if intrinsic_noise != 0: terms.append(ControlTerm(lambda t, x, args: vector_fields.f(t, x, args)*intrinsic_noise, VirtualBrownianTree(t0=t0-10**-5, t1=t1+10**-5, tol=1e-2, shape=(vector_fields.intrinsic_dim,), key=key)))
    if extrinsic_noise != 0: terms.append(ControlTerm(lambda t, x, args: lineax.DiagonalLinearOperator(jax.numpy.ones(x.shape[0])*extrinsic_noise), VirtualBrownianTree(t0=t0-10**-5, t1=t1+10**-5, tol=1e-2, shape=(vector_fields.dim,), key=key)))

    terms = MultiTerm(*tuple(terms))

    #adjoint=BacksolveAdjoint(solver=solver)
    adjoint=RecursiveCheckpointAdjoint(checkpoints=10)

    solution = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=0.01, y0=y0, saveat=saveat, max_steps=None, stepsize_controller=stepsize_controller,
                           adjoint=adjoint
                           )

    return solution
