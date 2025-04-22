import mdds.plotting.utils
from mdds.plotting import utils

import matplotlib
from matplotlib import pyplot as plt

import colorsys
import numpy as np

import warnings

trajectory_cmap = utils.shift_hue(utils.set_saturation(utils.set_lightness(matplotlib.colormaps['hsv_r'], 0.55), 0.9), -0.03)
manifold_cmap = matplotlib.colormaps['twilight']

light_grey = (0.5, 0.5, 0.5)
very_light_grey = (0.9, 0.9, 0.9)
very_dark_grey = (0.1, 0.1, 0.1)
dark_grey = (0.2, 0.2, 0.2)

color_theme = 'light'

match color_theme:
    case 'dark':
        background_color = very_dark_grey
        foreground_color = very_light_grey
        mid_foreground_color = very_light_grey
        mid_background_color = light_grey
    case 'light':
        background_color = (1, 1, 1)
        foreground_color = (0, 0, 0)
        mid_foreground_color = light_grey
        mid_background_color = very_light_grey


def get_pca_projection(xs, n=3):

    flat_xs = xs.reshape(-1, xs.shape[-1])

    U, S, V = np.linalg.svd(flat_xs, full_matrices=False)

    return V[:n].T


def integrate_controls(ts, xs, axis=1):

    temp = (xs[:, 1:]+xs[:, :-1])/2

    return np.add.accumulate(temp, axis=axis) * (ts[:, 1:, np.newaxis] - ts[:, :-1, np.newaxis])


def plot_var_exp(ax, test_frequency, ls, label='', color=foreground_color, linestyle='-', cla=True):

    if cla: ax.cla()

    utils.set_bottom_axis(ax, color=foreground_color)

    iterations = np.arange(len(ls))*test_frequency

    ax.plot(iterations, ls, color=color, linewidth=2.0, linestyle=linestyle,
            label=label+f'MDDS({iterations[-1]})=\n{ls[-1]:.3e}')

    ax.legend(loc=1, frameon=False)

    ax.set_xlabel('Iteration'), ax.set_ylabel('Var. Exp.')


def plot_var_exp_hline_pca(ax, data, dims, cmap=matplotlib.colormaps['Set2'], linestyle='--', cla=True):

    if cla: ax.cla()

    projection = get_pca_projection(data, n=max(dims))

    for di, d in enumerate(dims):
        l = 1 - ((data - data @ projection[:, :d] @ projection[:, :d].T)**2).mean()/np.var(data)
        ax.axhline(l, linestyle=linestyle, color=cmap(di), label=f'{d} PCs')


def remove_warning(f):

    def f_(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return f(*args, **kwargs)

    return f_


@remove_warning
def nanmean(*args, **kwarg): return np.nanmean(*args, **kwarg)

@remove_warning
def nanvar(*args, **kwarg): return np.nanvar(*args, **kwarg)


def plot_var_exp_hline_cond_avg(ax, data, condition, color=(1.0, 1.0, 0.1), linestyle='--', cla=True):

    if cla: ax.cla()

    data_sorted_per_cond = nanmean([nanmean((data[condition == c] - nanmean(data[condition == c], axis=0)) ** 2) for c in np.unique(condition)])

    ax.axhline(1-data_sorted_per_cond/nanvar(data), linestyle=linestyle, color=color, label='Cond. Avg')


def plot_trajectories(ax, xs, condition, trials_plotted=None, cmap=trajectory_cmap, alpha=1.0, linestyle='-', linewidth=1.0, zorder=3, cla=True):

    if cla: ax.cla()

    trials_plotted = xs.shape[0] if trials_plotted is None else trials_plotted

    for trial in range(0, xs.shape[0], xs.shape[0]//trials_plotted):
        ax.plot(xs[trial, :, 0], xs[trial, :, 1], xs[trial, :, 2], zorder=zorder,
                color=cmap(condition[trial]), alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        '''ax.scatter(xs[trial, :, 0], xs[trial, :, 1], xs[trial, :, 2], zorder=zorder+1,
                color=cmap(trial/xs.shape[0]), alpha=0.3)'''

    #utils.set_equal_lim(ax)
    ax.set_box_aspect((ax.get_xlim()[1]-ax.get_xlim()[0],
                       ax.get_ylim()[1]-ax.get_ylim()[0],
                       ax.get_zlim()[1]-ax.get_zlim()[0]))

    ax.set_xlabel(r'$\mathregular{x_1}$'), ax.set_ylabel(r'$\mathregular{x_2}$'), ax.set_zlabel(r'$\mathregular{x_3}$')


def match_axes_lim(ax, ax_template):

    ax.set_xlim(*ax_template.get_xlim())
    ax.set_ylim(*ax_template.get_ylim())
    ax.set_zlim(*ax_template.get_zlim())


def set_box_aspect(ax):
    ax.set_box_aspect((ax.get_xlim()[1] - ax.get_xlim()[0],
                       ax.get_ylim()[1] - ax.get_ylim()[0],
                       ax.get_zlim()[1] - ax.get_zlim()[0]))


def normalize(ts, min_t, max_t):

    ts = (ts - np.min(ts))
    ts = ts/(np.max(ts) - np.min(ts))
    ts = ts * (max_t - min_t)
    ts = ts + min_t

    return ts


def plot_manifold(ax, xs, cmap=manifold_cmap,
                       alpha=0.5, linewidth=2.0, edgecolor=(0.2, 0.2, 0.2),
                        gridlines=True,
                       ccount=6, rcount=6,
                       zorder=2, scatter=False, cla=True):

    if cla: ax.cla()

    min_alpha, max_alpha = 0.0, 0.5
    alpha_decay = 1 - np.exp(np.linspace(-5, 0, xs.shape[0]))
    alpha_decay = normalize(alpha_decay, min_alpha, max_alpha)

    gradient_ts = np.linspace(0, 1, xs.shape[1])
    colors = cmap(np.tile(gradient_ts, (xs.shape[0], 1)))
    colors[..., 3] = np.tile(alpha_decay, (xs.shape[1], 1))
    #colors[..., 3] = 0.2

    # ===== Surface =====
    '''ax.plot_surface(xs[..., 0], xs[..., 1], xs[..., 2],
                    facecolors=colors,
                    linewidth=0.02,
                    ccount=xs.shape[0], rcount=xs.shape[1],
                    shade=False,
                    zorder=zorder)'''

    if scatter:
        colors = cmap(np.tile(gradient_ts, (xs.shape[0], 1))).reshape(-1, 4)
        colors[..., 3] = 0.1
        ax.scatter(xs[..., 0].reshape(-1), xs[..., 1].reshape(-1), xs[..., 2].reshape(-1),
                        c=colors,
                        edgecolor=None,
                        s=100,
                        zorder=zorder)

    else:
        ax.plot_surface(xs[..., 0], xs[..., 1], xs[..., 2],
                        facecolors=colors,
                        linewidth=0.02,
                        ccount=xs.shape[0], rcount=xs.shape[1],
                        shade=False,
                        #alpha=0.1,
                        zorder=zorder)

    if gridlines:
        # ===== Grid lines =====
        min_alpha, max_alpha = 0.0, 1.0
        alpha_decay = 1 - np.exp(np.linspace(-5, 0, xs.shape[0]))
        alpha_decay = normalize(alpha_decay, min_alpha, max_alpha)

        #colors[..., 3] = np.tile(alpha_decay, (xs.shape[1], 1))

        cmap = utils.set_saturation(cmap, 0.9)
        gradient_ts = np.linspace(0, 1, xs.shape[0])
        colors = cmap(np.tile(gradient_ts, (xs.shape[1], 1)))
        colors[..., 3] = np.tile(alpha_decay, (xs.shape[1], 1))

        cmap_alpha = utils.get_cmap_interpolated(*colors[0, :])
        # ====== Lines =====
        for i in range(0, xs.shape[0], xs.shape[0]//rcount):
            utils.plot_with_gradient_3d(ax, xs[i, :, 0], xs[i, :, 1], xs[i, :, 2],
                                        gradient=gradient_ts, cmap=cmap_alpha, set_lim=False, zorder=zorder+0.5, alpha=None)

        #cmap_alpha = utils.get_cmap_interpolated(*colors[0, :])
        # ====== Circle =====
        for i in range(0, xs.shape[1], xs.shape[1]//ccount):
            #ax.plot(xs[:, i, 0], xs[:, i, 1], xs[:, i, 2], zorder=zorder+0.5, color=colors[0, i])
            cmap_ = mdds.plotting.utils.get_cmap_interpolated(colors[0, i], colors[0, i])
            utils.plot_with_gradient_3d(ax, xs[:, i, 0], xs[:, i, 1], xs[:, i, 2],
                                        gradient=np.ones(xs.shape[0]), cmap=cmap_, set_lim=False, zorder=zorder + 0.5,
                                        alpha=None)
        #ax.plot(xs[:, -1, 0], xs[:, -1, 1], xs[:, -1, 2], zorder=zorder+0.5, color=colors[0, -1, :3])


def add_vertical_axes(ax, number_axes, tick_size=8, spacing=0.15):

    axes = [ax.inset_axes([0,
                           (axi/number_axes)+(1/number_axes)*spacing,
                           1,
                           (1/number_axes)*(1-2*spacing)]) for axi in range(number_axes)][::-1]

    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for ax_i in axes:
        ax_i.set_facecolor(background_color)
        ax_i.tick_params(axis='both', labelsize=tick_size)

    return axes


def add_grid_axes(ax, number_columns, number_rows, tick_size=8, spacing=0.15):

    axes = [[ax.inset_axes([(axj/number_columns)+(1/number_columns)*spacing,
                           (axi/number_rows)+(1/number_rows)*spacing,
                           (1/number_columns)*(1-2*spacing),
                           (1/number_rows)*(1-2*spacing)]) for axj in range(number_columns)] for axi in range(number_rows)][::-1]

    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for ax_j in axes:
        for ax_i in ax_j:
            ax_i.set_facecolor(background_color)
            ax_i.tick_params(axis='both', labelsize=tick_size)

    return axes


def plot_jac(axs, J, cla=True):

    max_ = np.nanmax(np.abs(J))

    J = np.array(J)

    J[np.abs(J) < 10**-4] = 0

    for ax in axs:
        if cla: ax.cla()
        ax.axis('off')

    for i in range(J.shape[0]):
        im = axs[i].imshow(J[i], cmap=matplotlib.colormaps['seismic'], vmin=-max_, vmax=max_)


def plot_eigs(axs, Js, cla=True):

    for ax in axs:
        if cla: ax.cla()

    for i in range(Js.shape[0]):
        L = np.linalg.eigvals(Js[i])
        axs[i].scatter(L.real, L.imag, color=foreground_color, s=20, alpha=0.5, zorder=5)
        utils.set_centered_axes(axs[i], color=mid_background_color)

    for ax in axs[Js.shape[0]:]:
        ax.axis('off')


def plot_time_series(axs, ts, xs, condition, trials_plotted=6, cmap=trajectory_cmap, alpha=1.0, linestyle='-', linewidth=1.0,
                     variable_name='x', common_lim=True, cla=True):

    trials_plotted = min(xs.shape[0], trials_plotted)

    if cla:
        for ax in axs: ax.cla()

    for neuron in range(min(len(axs), xs.shape[-1])):
        for trial in range(0, xs.shape[0], xs.shape[0]//trials_plotted):
            axs[neuron].plot(ts[trial], xs[trial, :, neuron], color=cmap(condition[trial]), alpha=alpha, linestyle=linestyle, linewidth=linewidth)

        axs[neuron].set_ylabel(r'$\mathregular{'+variable_name+'_{'+str(neuron+1)+'}}$')
        utils.set_bottom_axis(axs[neuron], color=foreground_color)

    axs[min(len(axs), xs.shape[-1])-1].set_xlabel(r'$\mathregular{t}$')

    for ax in axs[min(len(axs), xs.shape[-1]):]: ax.axis('off')

    if common_lim:
        temp = xs[np.array(range(0, xs.shape[0], xs.shape[0] // trials_plotted)), :, :len(axs)]
        x_min, x_max = np.min(temp), np.max(temp)
        for ax in axs:
            ax.set_ylim(x_min, x_max)


def plot_2d_controls(ax, xs, condition, trials_plotted=100, cmap=trajectory_cmap, alpha=1.0, linewidth=1.0, linestyle='-', label=True, cla=True):

    if cla: ax.cla()

    utils.remove_axes(ax)

    for trial in range(0, xs.shape[0], max(1, xs.shape[0]//trials_plotted)):
        ax.plot(xs[trial, :, 0], xs[trial, :, 1], color=cmap(condition[trial]), alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.scatter(xs[trial, 0, 0], xs[trial, 0, 1], color=cmap(condition[trial]), alpha=alpha, s=5)
        #ax.scatter(xs[trial, -1, 0], xs[trial, -1, 1], color=cmap(condition[trial]), alpha=alpha, s=5)

    #utils.set_equal_lim(ax)

    if label:
        ax.set_xlabel(r'$\mathregular{\int c_1}$'), ax.set_ylabel(r'$\mathregular{\int c_2}$')

def plot_vector_fields(ax, xs, vfs, cmap=matplotlib.colormaps['Set2']):

    xs.reshape(-1, xs.shape[-1])

    for i, vf in enumerate(vfs.transpose(2, 0, 1, 3)):
        xs_ = xs.reshape(-1, xs.shape[-1]).T
        vf_ = vf.reshape(-1, vf.shape[-1]).T*0.1
        vf_ = np.stack([xs_, xs_ + vf_], axis=-2)
        for vf in vf_.transpose(2, 0, 1):
            ax.plot(*vf, color=cmap(i), linewidth=1.5)


def get_axes(cols, rows, color_background, computed_zorder=True):

    fig = plt.figure(figsize=(cols*4,rows*4), constrained_layout=True, dpi=70)
    gs = fig.add_gridspec(ncols=cols,nrows=rows)
    axs_3d = []
    axs_ignored = []
    for i in range(2,4):
        for j in range(0,2):
            axs_ignored.append([i,j])
    for j in range(4):
        axs_ignored.append([4, j])
    axs_3d = axs_3d + [[1,0]]
    axs = np.array([[fig.add_subplot(gs[i,j], projection=('3d' if [j,i] in axs_3d else None), **({'computed_zorder': computed_zorder} if [j,i] in axs_3d else {})) if [j,i] not in axs_ignored else None for i in range(rows)] for j in range(cols)])

    ax_giga = fig.add_subplot(gs[0:2,2:4], computed_zorder=computed_zorder, projection='3d')
    ax_time_series = fig.add_subplot(gs[:, 4])

    for ax_ in axs:
        for ax in ax_:
            if ax is not None: ax.set_facecolor(color_background)
    fig.set_facecolor(color_background)
    ax_giga.set_facecolor(color_background)

    for ax_ in axs:
        for ax in ax_:
            if hasattr(ax, 'get_zlim'):
                utils.set_pannels_3d(ax, True, True, True, mid_background_color, foreground_color)
    utils.set_pannels_3d(ax_giga, True, True, True, mid_background_color, foreground_color)

    return fig, axs, ax_giga, ax_time_series
