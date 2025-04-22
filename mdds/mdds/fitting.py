import numpy as np

from .build import *
from mdds.plotting.plotting import *
from .initialization import initialize
from .utils import reset_controls, decoder_vmap, split_model, load_model
#from .loss import loss, loss_grad
from .projected_grad.loss import loss, loss_grad, loss_
from .projected_grad.grad import tree_map, project_grad

from mdds.controls import get_grid_hypersphere, evaluate_control, batch_controls

# ===== To be dealt with =====
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from jax import numpy as jnp
from jax import random
import jax

import equinox as eqx

from tqdm.auto import tqdm
import pickle
from math import ceil
import scipy


def is_notebook():
    import __main__ as main
    return not hasattr(main, '__file__')


if is_notebook():
    from IPython.display import display, clear_output
    from tqdm import tqdm


def nanmean(*args, **kwarg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(*args, **kwarg)


def fit(data, condition, manifold_dim, embedding_dim,
        data_grid=None,
        controls_interpolation_freq=1.0, vector_fields_class='OrthogonalMDDS', vector_fields_hyperparameters={}, optimized=('controls', 'vector_fields', 'decoder'),
        frobenius_penalty=10, learning_rate=0.5e-3, weight_decay=1.0, batch_prop=0.5, max_iterations=10**5,
        freq_reset_controls=50, min_std_reset_controls=1e-2,
        test_frequency=10**2, min_test_iterations=10, min_std_test_loss=1e-4, mesh_size=91,
        rtol=1e-5, atol=1e-6, intrinsic_noise=0.0, extrinsic_noise=0.0,
        cmap_manifold=matplotlib.colormaps['twilight'], cmap_condition=matplotlib.colormaps['hsv'],
        load_dir='.', save_dir=None, seed=0, *args, **kwargs):

    eqx.clear_caches()
    jax.clear_caches()  # Necessary to avoid bug in diffrax/eqx

    key = random.PRNGKey(seed)

    trial_dim, time_dim, neuron_dim = tuple(data.shape)
    t0, t0_data, t1 = -0.5, 0.0, 1

    data_time_mask = ~np.isnan(data)[..., 0]#True if not np.isnan(data).any() else

    control_time_dim = int(controls_interpolation_freq*time_dim*(t1-t0))
    control_ts = jnp.linspace(t0, t1, control_time_dim)

    # ===== Model =====
    key, model_key, decoder_key = random.split(key, 3)
    model = build_model(t0, t1, trial_dim, neuron_dim, embedding_dim, manifold_dim, model_key, control_ts,
                        vector_fields_class, vector_fields_hyperparameters, data,
                        None if embedding_dim == data.shape[-1] else decoder_key)

    #key, subkey = random.split(key)
    #model = initialize(subkey, model, jnp.linspace(t0_data, t1, time_dim), data)

    # ===== Other stuff =====
    model_diff, model_static = split_model(model, optimized)
    optimizer, opt_state = build_optimizer(model_diff, learning_rate, weight_decay)
    solver, stepsize_controller = build_solver(rtol, atol, control_ts)

    saveat = jnp.stack([jnp.linspace(t0_data, t1, time_dim) for _ in range(trial_dim)])

    # ===== Loading from file =====
    model = load_model(model, load_dir, '/model.eqx', optimized)#, load_model(opt_state, load_dir, '/opt_state.eqx')

    # ===== To build inferred manifold =====
    saveat_test = jnp.stack([jnp.linspace(t0, t1, mesh_size) for _ in range(mesh_size)])

    controls_test = build_control(jnp.linspace(t0, t1, mesh_size), get_grid_hypersphere(mesh_size, manifold_dim)*1.5)

    # ===== Plot stuff =====
    utils.set_font(font_color=foreground_color, font_size=20)
    fig, axs, ax_giga, axs_factors = get_axes(5, 3, background_color)
    axs_time_series = add_vertical_axes(axs[1, 1], number_axes=min(4, neuron_dim))
    axs_factors = add_vertical_axes(axs_factors, number_axes=min(3, neuron_dim), tick_size=20)
    axs_2d_control = add_grid_axes(axs[0, 2], manifold_dim-1, manifold_dim-1, spacing=0.05) if manifold_dim>=3 else axs[0, 2].axis('off')
    axs_jac = [ax for ax_ in add_grid_axes(axs[1, 2], ceil(np.sqrt(manifold_dim)), ceil(np.sqrt(manifold_dim)), spacing=0.01) for ax in ax_]
    axs_jac_schur = [ax for ax_ in add_grid_axes(axs[2, 2], ceil(np.sqrt(manifold_dim)), ceil(np.sqrt(manifold_dim)), spacing=0.01) for ax in ax_]
    axs_jac_eig = [ax for ax_ in add_grid_axes(axs[3, 2], ceil(np.sqrt(manifold_dim)), ceil(np.sqrt(manifold_dim)), spacing=0.05) for ax in ax_]

    # ===== Training =====
    iterator = tqdm(range(max_iterations), position=0, desc="Training initialization ... ", unit=" iteration", ncols=300, leave=False)
    ls_test, ls_data_test, ls_lie_bracket_test = [], [], []

    reset_controls_flag = True

    data_start = np.argmax(data_time_mask, axis=1)
    increment = jax.vmap(random.randint, in_axes=(0, None, 0, None))(random.split(key, trial_dim), (1,), data_start, time_dim)[:, 0]
    max_time_ids_0 = data_start + 2000 #+ increment

    for iteration in iterator:

        # ===== Update time ids =====
        max_time_ids = np.clip(max_time_ids_0 + iteration//20, 0, time_dim)

        time_mask = np.zeros(data.shape[:2], dtype=bool)
        for i, tm in enumerate(time_mask): tm[data_start[i]:max_time_ids[i]] = True
        not_time_mask = ~time_mask
        for i, tm in enumerate(not_time_mask): tm[[data_start[i], max_time_ids[i]-1]] = True
        time_mask, not_time_mask = time_mask & data_time_mask, not_time_mask #& data_time_mask

        # ===== Reset some controls =====
        if iteration % freq_reset_controls == freq_reset_controls - 1 and reset_controls_flag and 'controls' in optimized:
            key, subkey = random.split(key)
            model = eqx.tree_at(lambda m: m['controls'], model, reset_controls(subkey, model['controls'], 1, False))
            #model['controls'] = reset_controls(subkey, model['controls'], 1, False)

        # ===== Test =====
        if iteration % test_frequency == 0:

            # ===== Evaluate on uniform (circular) controls =====
            subkey, key = random.split(key)
            l_test, (solution_test, l_data, l_lie_bracket) = loss(model, stepsize_controller, saveat, solver, t0,
                                                                  0.0, 0.0, data,
                                                                  jnp.arange(trial_dim).astype(int),
                                                                  jnp.arange(neuron_dim).astype(int),
                                                                  data_time_mask,
                                                                  0, subkey)

            ys = solution_test.ys

            ts, xs = solution_test.ts, decoder_vmap(model['decoder'], ys)
            xs_mask = np.where(time_mask[..., np.newaxis], xs, np.nan)
            xs_mask_not = np.where(not_time_mask[..., np.newaxis], xs, np.nan)

            l_data = jnp.mean((xs - data)[time_mask]**2)/jnp.var(data[time_mask])

            ls_test.append(l_test), ls_data_test.append(l_data), ls_lie_bracket_test.append(l_lie_bracket)

            # ===== Plotting ===== PSTH?
            unique_conditions = np.unique(condition)
            ts_first_cond = np.stack([ts[condition == c][0] for c in unique_conditions])
            with warnings.catch_warnings(): # Nanmean raises warning when taking mean over slice full of nans
                warnings.simplefilter("ignore", category=RuntimeWarning)
                plot_time_series(axs_time_series, ts_first_cond, np.stack([np.nanmean(xs_mask[condition == c], axis=0) for c in unique_conditions]), unique_conditions, common_lim=False, cmap=cmap_condition) # k trials per condition?
                #plot_time_series(axs_time_series, ts_first_cond, np.stack([xs_mask_not[condition == c][0] for c in unique_conditions]), unique_conditions, common_lim=False, cmap=cmap_condition, linestyle='dotted', alpha=0.2, cla=False) # k trials per condition?
                plot_time_series(axs_time_series, ts_first_cond, np.stack([np.nanmean(data[condition == c], axis=0) for c in unique_conditions]), unique_conditions, common_lim=False, linestyle='--', alpha=0.5, cla=False, cmap=cmap_condition)

            # ===== Plot controls 2D =====
            preparatory_steps = int(time_dim*(t0_data-t0))
            ts_control = jnp.tile(jnp.linspace(t0, t1, time_dim+preparatory_steps), (trial_dim, 1))
            cs = evaluate_control(batch_controls(model['controls'], jnp.arange(trial_dim)), ts_control)
            cs_integrated = integrate_controls(ts_control, cs)
            #cs_integrated_t0_data = cs_integrated[:, ts_control[:-1] >= t0_data]
            ts_control_temp = ts_control[:, :-1]#np.tile(ts_control[:-1], (trial_dim, 1))
            for time_mask_temp, alpha, linestyle, cla in zip([time_mask, not_time_mask], [1.0, 0.5], ['-', 'dotted'], [True, False]):
                time_mask_temp = np.concatenate([np.full((trial_dim, preparatory_steps), not cla), time_mask_temp], axis=1)
                time_mask_temp[:, preparatory_steps] = True
                cs_integrated_temp = np.where(time_mask_temp[:, :-1, np.newaxis], cs_integrated, np.nan)
                cs_integrated_avg = np.stack([nanmean(cs_integrated_temp[condition == c], axis=0) for c in unique_conditions])
                if manifold_dim >= 2:
                    plot_2d_controls(axs[0, 1], cs_integrated_temp[:, ts_control[0, :-1] >= t0_data], condition, alpha=0.5*alpha, linestyle=linestyle, cmap=cmap_condition, cla=cla) # Could plot k trial per condition
                    plot_2d_controls(axs[0, 1], cs_integrated_avg[:, ts_control[0, :-1] >= t0_data], unique_conditions, linewidth=2.5, linestyle=linestyle, alpha=alpha, cla=False, cmap=cmap_condition)
                if manifold_dim >= 3:
                    for i in range(manifold_dim):
                        for j in range(manifold_dim-1):
                            if i > j:
                                plot_2d_controls(axs_2d_control[i-1][j], cs_integrated_avg[:, ts_control[0, :-1] >= t0_data][..., [i, j]], unique_conditions, linewidth=1.0, linestyle=linestyle, alpha=alpha, label=False, cmap=cmap_condition, cla=cla)
                            else:
                                axs_2d_control[i][j].axis('off')
            # ===== Plot controls time series =====
                plot_time_series(axs_factors, ts_control_temp, cs_integrated_temp, condition, common_lim=False, cmap=cmap_condition, alpha=0.5*alpha, linestyle=linestyle, variable_name=r'\int c', cla=cla)
                plot_time_series(axs_factors, ts_control_temp, cs_integrated_avg, unique_conditions, common_lim=False, cmap=cmap_condition, linewidth=2.5, alpha=alpha, linestyle=linestyle, variable_name=r'\int c', cla=False)

            # ===== Plot trajectories =====
            projection = get_pca_projection(data[data_time_mask]) if neuron_dim != 3 else jnp.eye(neuron_dim)
            xs_per_condition = np.stack([nanmean(np.where(time_mask[..., np.newaxis], xs, np.nan)[condition == c], axis=0) for c in unique_conditions])
            data_per_condition = np.stack([nanmean(data[condition == c], axis=0) for c in unique_conditions])
            plot_trajectories(axs[1, 0], data_per_condition @ projection, unique_conditions, linestyle='--', alpha=0.5, cmap=cmap_condition)
            plot_trajectories(axs[1, 0], xs_per_condition @ projection, unique_conditions, cla=False, cmap=cmap_condition)
            if data_grid is not None: plot_manifold(axs[1, 0], data_grid @ projection, cla=False, cmap=cmap_manifold)

            # ===== Plot inferred manifold =====
            if manifold_dim >= 2:
                # ===== Normalize the controls for testing =====
                masked_saveat = jnp.where(time_mask, saveat, jnp.nan)
                ts_control_masked = jax.vmap(jnp.linspace, in_axes=(None, 0, None))(t0, jnp.nanmax(masked_saveat, axis=1), preparatory_steps+jnp.max(max_time_ids))#jnp.tile(jnp.linspace(t0, t1, time_dim+preparatory_steps), (mesh_size, 1))
                cs = evaluate_control(batch_controls(model['controls'], jnp.arange(trial_dim)), ts_control_masked)
                cs_integrated = integrate_controls(ts_control_masked, cs)
                max_c = jnp.max(jnp.abs(cs_integrated), axis=(0, 1))#, 0.99
                max_c = jnp.max(jnp.linalg.norm(cs_integrated, axis=-1))
                std_c = jnp.nanstd(jnp.linalg.norm(cs_integrated, axis=-1))
                mean_c = jnp.nanmean(jnp.linalg.norm(cs_integrated, axis=-1))
                mean_c = jnp.nanmean(model['controls'].interpolator.ys, axis=(0, 1))
                std_c = jnp.nanstd(model['controls'].interpolator.ys, axis=(0, 1))
                ts_control_test = jnp.linspace(t0, t1, time_dim+jnp.max(max_time_ids))#jnp.tile(jnp.linspace(t0, t1, time_dim+preparatory_steps), (mesh_size, 1))
                ts_control_test = jnp.tile(ts_control_test, (mesh_size, 1))
                cs_test = evaluate_control(controls_test, ts_control_test)
                cs_test_integrated = integrate_controls(ts_control_test, cs_test)
                max_c_test = jnp.nanmax(jnp.abs(cs_test_integrated), axis=(0, 1))
                max_c_test = jnp.nanmax(jnp.linalg.norm(cs_test_integrated, axis=-1))
                std_c_test = jnp.nanstd(jnp.linalg.norm(cs_test_integrated, axis=-1))
                mean_c_test = jnp.nanmean(jnp.linalg.norm(cs_test_integrated, axis=-1))
                #std_c_test = jnp.nanstd(controls_test.interpolator.ys, axis=(0, 1))
                mean_c_test = jnp.nanmean(controls_test.interpolator.ys, axis=(0, 1))
                c_test = (controls_test.interpolator.ys-mean_c_test)/std_c_test#/(t1 - t0) # this should be divided by the integral
                #print(max_c_test, jnp.mean(c_test, axis=(0, 1)), np.std(c_test, axis=(0, 1)), jnp.max(c_test, axis=(0, 1)))
                #controls_test = eqx.tree_at(lambda controls_: controls_.interpolator.ys, controls_test, 0.5*c_test*std_c+mean_c)

                '''max_c = jnp.quantile(jnp.abs(cs_integrated), 0.99, axis=(0, 1))
                c_test = controls_test.interpolator.ys / jnp.max(jnp.abs(controls_test.interpolator.ys), axis=(0, 1)) / (t1 - t0)
                controls_test = eqx.tree_at(lambda controls_: controls_.interpolator.ys, controls_test, 0.02*c_test * max_c)'''

                # ===== Evaluate for testing =====
                _, (solution_manifold, _, _) = loss(split_model(model, ['controls'])[1] | {'controls': controls_test},
                                             PIDController(rtol, atol, dtmax=0.01, dtmin=0.001), saveat_test, solver, t0, 0.0, 0.0,
                                             jnp.zeros((mesh_size, mesh_size, neuron_dim)),
                                             jnp.arange(mesh_size).astype(int), jnp.arange(neuron_dim).astype(int), jnp.ones(saveat_test.shape[0:2], dtype=bool), 0, subkey)
                xs_manifold = decoder_vmap(model['decoder'], solution_manifold.ys)

                '''import umap
                um = umap.UMAP(n_components=3, n_neighbors=150)
                #um.fit(xs_manifold.reshape(-1, xs_manifold.shape[-1]))
                um.fit(data[::5].reshape(-1, xs_manifold.shape[-1]))
                xs_manifold_pcs = um.transform(xs_manifold.reshape(-1, xs_manifold.shape[-1])).reshape(list(xs_manifold.shape[:2])+[3])'''
                xs_manifold_pcs = xs_manifold @ projection

                #projection = get_pca_projection(ys_manifold) if embedding_dim != 3 else jnp.eye(neuron_dim)
                plot_manifold(ax_giga, xs_manifold_pcs, cmap=cmap_manifold)

                # ===== Vector field plot =====
                ys_vfs = solution_manifold.ys[::solution_manifold.ys.shape[0]//12, solution_manifold.ys.shape[0]//5::solution_manifold.ys.shape[0]//5]
                vfs = eqx.filter_vmap(eqx.filter_vmap(model['vector_fields'].F))(ys_vfs)
                dec = lambda x: model['decoder'](x[jnp.newaxis, ...], slice(None))[0]
                xs_vfs, vfs = jax.vmap(jax.vmap(jax.vmap(jax.jvp, in_axes=(None, None, -1)), in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(dec, (ys_vfs,), (vfs,))
                xs_vfs = xs_vfs[..., 0, :]
                '''vfs = np.array(vfs) ; vfs[..., -1] = 0.0
                xs_vfs = np.array(xs_vfs) ; xs_vfs[..., -1] = 0.5'''
                #xs_vfs, vfs = ys_vfs, vfs.swapaxes(-2, -1)
                xs_vfs, vfs = xs_vfs @ projection, vfs @ projection
                plot_vector_fields(ax_giga, xs_vfs, vfs)

                #for time_mask_temp, alpha, linestyle in zip([time_mask, not_time_mask], [1.0, 0.5], ['-', 'dotted']):
                for time_mask_temp, alpha, linestyle in zip([time_mask], [1.0], ['-']):
                    xs_temp = xs#np.where(time_mask_temp[..., np.newaxis], xs, np.nan)
                    xs_per_condition = np.stack([xs_temp[condition == c][0] for c in unique_conditions])
                    #temp = um.transform(xs_per_condition.reshape(-1, xs_per_condition.shape[-1])).reshape(list(xs_per_condition.shape[:2])+[3])
                    temp = xs_per_condition @ projection
                    plot_trajectories(ax_giga, temp, unique_conditions, alpha=alpha, linestyle=linestyle, cla=False, cmap=cmap_condition)

                #match_axes_lim(ax_giga, axs[1, 0])
                set_box_aspect(ax_giga)

            # ===== Jacobian / Weights =====
            Js = (jax.jacrev(model['vector_fields'].F)(model['vector_fields'].get_initial_state())/jnp.linalg.norm(model['vector_fields'].F(model['vector_fields'].get_initial_state()), axis=0)[jnp.newaxis, :, jnp.newaxis]).transpose(1, 0, 2)
            #Js = model['vector_fields'].Ws
            Js_schur = np.stack([scipy.linalg.schur(J)[0] for J in Js]) + np.tril(np.full((embedding_dim, embedding_dim), np.nan), k=-2)

            plot_jac(axs_jac, Js)
            plot_jac(axs_jac_schur, Js_schur)
            plot_eigs(axs_jac_eig, Js)

            # ===== Loss =====
            var_exp = 1-np.array(ls_data_test)
            plot_var_exp(axs[0, 0], test_frequency, var_exp)
            plot_var_exp_hline_pca(axs[0, 0], data[data_time_mask], [manifold_dim, embedding_dim+1], cla=False)
            plot_var_exp_hline_cond_avg(axs[0, 0], data, condition, cla=False)
            axs[0, 0].legend(frameon=False), axs[0, 0].set_ylim(0, min(axs[0, 0].get_ylim()[1], 1.05))

            # ===== Save model =====
            if save_dir is not None:
                #with open(save_dir + '/mdds_'+save_dir.split('/')[-1]+'.pkl', 'wb') as f: pickle.dump(fig, f)
                plt.savefig(save_dir + '/mdds_'+save_dir.split('/')[-1]+'.pdf')
                np.savetxt(save_dir + '/variance_explained.csv', var_exp, delimiter=',')
                np.save(save_dir+'/data_estimate.npy', xs)
                np.save(save_dir+'/latent_factors.npy', cs_integrated[:, -time_dim:])
                np.save(save_dir+'/ambient_factors.npy', ys)
                np.save(save_dir+'/Js.npy', Js)
                if manifold_dim>=2:
                    np.save(save_dir+'/manifold.npy', xs_manifold)
                    np.save(save_dir+'/xs_vfs.npy', xs_vfs)
                    np.save(save_dir+'/vfs.npy', vfs)
                eqx.tree_serialise_leaves(save_dir + '/model.eqx', model)
                eqx.tree_serialise_leaves(save_dir + '/opt_state.eqx', opt_state)

            # ===== Plot model =====
            if is_notebook():
                clear_output()
                display(fig)
            else:
                fig.canvas.draw()
                plt.pause(3.0)

            current_std = jnp.std(jnp.array(ls_test[-min_test_iterations:]))
            if len(ls_test) >= min_test_iterations and current_std < min_std_reset_controls:
                reset_controls_flag = False

            # ===== Printing =====
            if len(ls_test) >= min_test_iterations and current_std < min_std_test_loss:
                iterator.set_description(f'Final loss -- Loss train: {l:.5e},  Loss test: {l_test:.5e}, Loss data: {l_data:.5e}, Loss Lie Bracket: {l_lie_bracket:.5e}, Reset controls: {reset_controls_flag}')
                break

        '''from functools import partial

        value_fn = partial(loss_,
                stepsize_controller=stepsize_controller,
                saveat=saveat, solver=solver, t0=t0,
                max_steps_solver=max_steps_solver, data=data,
                trial_batch_ids=jnp.arange(trial_dim).astype(int),
                neuron_batch_ids=jnp.arange(neuron_dim).astype(int),
                time_mask=data_time_mask,
                frobenius_penalty=0.0, key=subkey)


        @eqx.filter_jit
        def value_fn_filtered(model, args):

            static, model_static = args

            model_diff = eqx.combine(model, static)

            model = model_diff | model_static

            return value_fn(model)

        import optimistix as optx

        # Set up the solver
        solver_ = optx.BFGS(rtol=1e-3, atol=1e-3)
        solver_ = optx.BestSoFarLeastSquares(solver_)

        model_diff, model_static = split_model(model, ['controls'])
        params, static = eqx.partition(model_diff, eqx.is_inexact_array)

        # Minimize the loss function
        sol = optx.least_squares(value_fn_filtered, solver_, params, args=(static, model_static), max_steps=2 ** 14)#, options={'jac': 'bwd'}
        params = sol.value

        #print(sol.stats)
        #l_test, current_std = 0, 0

        #print('AAAAAAA')

        model_diff = eqx.combine(params, static)
        model = model_diff | model_static'''

        # =======

        '''model_diff, model_static = split_model(model, ['controls'])
        params, static = eqx.partition(model_diff, eqx.is_inexact_array)

        # Minimize the loss function
        sol = optx.minimise(value_fn_filtered, solver_, params, args=(static, model_static), max_steps=2 ** 14)
        params = sol.value

        #print(sol.stats)
        #l_test, current_std = 0, 0

        #print('AAAAAAA')

        model_diff = eqx.combine(params, static)
        model = model_diff | model_static'''

        # ===== We fit both the controls and manifold on this batch =====
        key, key_trial, key_neuron, key_time = random.split(key, 4)
        batch_ids = jnp.sort(random.choice(key_trial, trial_dim, (ceil(trial_dim*batch_prop),), replace=False))
        #batch_ids = jnp.arange(trial_dim, dtype=int)
        #neuron_batch_ids = jnp.sort(random.choice(key_neuron, neuron_dim, (ceil(neuron_dim*batch_prop),), replace=False))
        neuron_batch_ids = jnp.arange(neuron_dim, dtype=int)

        # ===== Fit manifold =====
        model_diff, model_static = split_model(model, optimized)
        key, subkey = random.split(key)
        (l, (solution, l_data, l_lie_bracket)), grads, grads_constraint = loss_grad(model_diff, model_static,
                                                                        stepsize_controller, saveat, solver, t0,
                                                                              intrinsic_noise, extrinsic_noise, data,
                                                                              batch_ids, neuron_batch_ids,
                                                                              time_mask, frobenius_penalty, subkey)

        # ===== Update =====
        grads = tree_map(project_grad, grads, grads_constraint, frobenius_penalty) #if (iteration // 500) % 2 == 0  else tree_map(project_grad, grads_constraint, grads)
        #grads = eqx.tree_at(lambda g: g['vector_fields'], grads, jax.tree_util.tree_map(lambda x: 100000*x, grads['vector_fields']))
        updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), eqx.filter(opt_state, eqx.is_array), eqx.filter(model_diff, eqx.is_array))
        '''updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), eqx.filter(opt_state, eqx.is_array), eqx.filter(model_diff, eqx.is_array), value=l, grad=grads, value_fn=partial(loss_,
                                                                                                                  stepsize_controller=stepsize_controller,
                                                                                                                  saveat=saveat, solver=solver, t0=t0,
                                                                                                                  max_steps_solver=max_steps_solver, data=data,
                                                                                                                  trial_batch_ids=jnp.arange(trial_dim).astype(int),
                                                                                                                  neuron_batch_ids=jnp.arange(neuron_dim).astype(int),
                                                                                                                  time_mask=data_time_mask,
                                                                                                                  frobenius_penalty=0, key=subkey), linesearch=False)'''
        '''if l_lie_bracket>10**-4:
            updates = jax.tree_map(lambda x, y: project_grad_rectify(x, -y), updates, grads_constraint)
        else:
            updates = jax.tree_map(lambda x, y: project_grad(x, -y), updates, grads_constraint)'''
        model_diff = eqx.apply_updates(model_diff, updates)
        model = model_diff | model_static

        iterator.set_description(f'Loss train: {l:.4e},  Loss test: {l_test:.4e}, Loss Lie Bracket: {l_lie_bracket:.5e}, Loss data: {l_data:.5e}, Reset controls: {reset_controls_flag}, Std: {current_std:.4e}/{min_std_test_loss:.1e}')
