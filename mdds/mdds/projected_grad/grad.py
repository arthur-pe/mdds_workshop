import jax, jax.numpy as jnp
import equinox as eqx


def value_grad_aux(f):

    def f_(x, *args):

        f_partial = lambda x: f(x, *args)

        f_x, f_vjp, aux = eqx.filter_vjp(f_partial, x, has_aux=True)

        jac = tuple([f_vjp(tuple(jnp.eye(len(f_x))[i]))[0] for i in range(len(f_x))])

        return (f_x[0], aux), jac

    return f_


def value_projected_grad_aux(f):

    def f_(x, *args):

        (f_x, aux), jac = value_grad_aux(f)(x, *args)

        #projected_grads = tree_map(project_grad, jac[0], jac[1])

        return (f_x, aux), jac[0], jac[1]

    return f_


def tree_map(func, *args):

    temp = list(zip(*[jax.flatten_util.ravel_pytree(a) for a in args]))

    return temp[1][0](func(*temp[0]))


def norm(x): return jnp.sqrt(jnp.sum(jnp.square(x)))

def clip_by_norm(grad, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))
    clip_coef = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_map(lambda g: g * clip_coef, grad)

@eqx.filter_jit
def project_grad(gradient_optimized, gradient_constraint, ratio=0.9):

    temp = gradient_optimized + ratio * gradient_constraint

    #temp = clip_by_norm(temp, 1.0)

    return temp #+ ratio*gradient_constraint

    f_min = jnp.finfo(gradient_optimized).eps

    normalized_gradient_constraint = gradient_constraint/(norm(gradient_constraint) + f_min)

    projection_of_grad_opt = jnp.sum(gradient_optimized * normalized_gradient_constraint)*normalized_gradient_constraint

    grad_optimized_orth_constraint = gradient_optimized - projection_of_grad_opt

    normalized_grad_optimized_orth_constraint = grad_optimized_orth_constraint/(norm(grad_optimized_orth_constraint) + f_min)

    normalized_gradient_constraint = gradient_constraint / (norm(gradient_constraint) + f_min)

    return normalized_grad_optimized_orth_constraint*norm(gradient_optimized) + ratio*gradient_constraint

    return (ratio*normalized_grad_optimized_orth_constraint+(1-ratio)*normalized_gradient_constraint)*(ratio*norm(gradient_optimized) + (1-ratio)*norm(gradient_constraint))

    return normalized_grad_optimized_orth_constraint*norm(gradient_optimized)


@eqx.filter_jit
def project_grad_rectify(gradient_optimized, gradient_constraint):

    return gradient_constraint

    f_min = jnp.finfo(gradient_optimized).eps

    normalized_gradient_constraint = gradient_constraint/(norm(gradient_constraint) + f_min)

    projection_of_grad_opt = jnp.sum(gradient_optimized * normalized_gradient_constraint)*normalized_gradient_constraint

    grad_optimized_orth_constraint = (gradient_optimized - projection_of_grad_opt)
    normalized_grad_optimized_orth_constraint = grad_optimized_orth_constraint/(norm(grad_optimized_orth_constraint) + f_min)

    #return gradient_constraint*0.01+normalized_grad_optimized_orth_constraint*jnp.linalg.norm(gradient_optimized)
    new_grad = (0.5*normalized_gradient_constraint+0.5*normalized_grad_optimized_orth_constraint)*norm(gradient_optimized)
    return new_grad
