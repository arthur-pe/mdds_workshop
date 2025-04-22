import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import matplotlib
import os
import sys
import colorsys
import math

import mdds

default_cmap = matplotlib.colormaps['hsv']
axes_line_width = 1.5
default_font_size = 14


def plot_with_gradient_3d(ax, xs, ys, zs, gradient, cmap,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          linewidth=1.0, alpha=1.0, zorder=3, set_lim=True):

    points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    number_segments = len(segments)

    if dashed:
        temp = []
        temp_cols = []
        for i in range(number_dashes):
            temp.append(segments[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
            temp_cols.append(gradient[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])

        segments = np.concatenate(temp)
        gradient = np.concatenate(temp_cols)

    capstyle = 'round' if dashed or alpha==1 else 'butt'
    lc = Line3DCollection(segments, cmap=cmap, alpha=alpha, zorder=zorder, capstyle=capstyle)
    lc.set_array(gradient)
    lc.set_clim(0,1)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)

    if set_lim:
        ax.set_xlim(min((np.min(xs), ax.get_xlim()[0])), max((np.max(xs), ax.get_xlim()[1])))
        ax.set_ylim(min((np.min(ys), ax.get_ylim()[0])), max((np.max(ys), ax.get_ylim()[1])))
        ax.set_zlim(min((np.min(zs), ax.get_zlim()[0])), max((np.max(zs), ax.get_zlim()[1])))


def plot_with_gradient_2d(ax, xs, ys, gradient, cmap,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          linewidth=1.0, alpha=1.0, zorder=3, set_lim=True):

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    number_segments = len(segments)

    if dashed:
        temp = []
        temp_cols = []
        for i in range(number_dashes):
            temp.append(segments[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
            temp_cols.append(gradient[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
        segments = np.concatenate(temp)
        gradient = np.concatenate(temp_cols)

    capstyle = 'round' if dashed or alpha==1 else 'butt'
    lc = LineCollection(segments, cmap=cmap, alpha=alpha, zorder=zorder, capstyle=capstyle)
    lc.set_array(gradient)
    lc.set_clim(0,1)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)

    if set_lim:
        ax.set_xlim(min((np.min(xs), ax.get_xlim()[0])), max((np.max(xs), ax.get_xlim()[1])))
        ax.set_ylim(min((np.min(ys), ax.get_ylim()[0])), max((np.max(ys), ax.get_ylim()[1])))


def hls_to_rgb(colors): return np.array([colorsys.hls_to_rgb(*c[:3]) for c in colors])
def rgb_to_hls(colors): return np.array([colorsys.rgb_to_hls(*c[:3]) for c in colors])
def sigmoid(x): return 1/(1+np.exp(-x))


def set_lightness(cmap, lightness, interpolation_points=9):

    colors = cmap(np.linspace(0, 1, interpolation_points))

    colors = rgb_to_hls(colors)

    colors = np.stack([colors[:,0], np.full(interpolation_points, lightness), colors[:,2]], axis=-1)

    return get_cmap_interpolated(*hls_to_rgb(colors))


def set_saturation(cmap, saturation, interpolation_points=9):

    colors = cmap(np.linspace(0, 1, interpolation_points))

    colors = rgb_to_hls(colors)

    colors = np.stack([colors[:,0], colors[:,1], np.full(interpolation_points, saturation)], axis=-1)

    return get_cmap_interpolated(*hls_to_rgb(colors))


def increase_lightness(cmap, d_lightness, interpolation_points=9):

    colors = cmap(np.linspace(0, 1, interpolation_points))

    colors = rgb_to_hls(colors)

    colors = np.stack([colors[:, 0], np.clip(colors[:, 1] + d_lightness, 0.0, 1.0), colors[:, 2]], axis=-1)

    return get_cmap_interpolated(*hls_to_rgb(colors))


def increase_saturation(cmap, d_saturation, interpolation_points=9):

    colors = cmap(np.linspace(0, 1, interpolation_points))

    colors = rgb_to_hls(colors)

    colors = np.stack([colors[:, 0], colors[:, 1], np.clip(colors[:, 2] + d_saturation, 0.0, 1.0)], axis=-1)

    return get_cmap_interpolated(*hls_to_rgb(colors))


def shift_hue(cmap, hue_shift, interpolation_points=9):

    colors = cmap(np.linspace(0, 1, interpolation_points))

    colors = rgb_to_hls(colors)

    temp = (colors[:, 0]+hue_shift)
    temp[temp>1] = 1-temp[temp>1]
    temp[temp<0] = temp[temp<0]+1

    colors = np.stack([temp, colors[:,1], colors[:,2]], axis=-1)

    return get_cmap_interpolated(*hls_to_rgb(colors))


def get_cmap_interpolated(*args):

    colors = []
    for i in range(len(args)-1):
        colors.append(np.stack([np.linspace(args[i][0],args[i+1][0],1001),
                                np.linspace(args[i][1],args[i+1][1],1001),
                                np.linspace(args[i][2],args[i+1][2],1001),
                                np.linspace(args[i][3] if len(args[i]) == 4 else 1,
                                            args[i+1][3] if len(args[i+1]) == 4 else 1, 1001)], axis=-1))
    colors = np.concatenate(colors, axis=0)
    cmap = LinearSegmentedColormap.from_list('interpolated_cmap', colors)

    return cmap


def get_cmap_black(color):

    return get_cmap_interpolated((0,0,0,1), color)


def get_cmap_white(color):

    return get_cmap_interpolated((1,1,1,1), color)


def set_pannels_3d(ax, x=False, y=False, z=True, panels_color=(0.95, 0.95, 0.95), grid_color=(0.1, 0.1, 0.1)):

    c_transparent = np.array([1, 1, 1, 0])

    ax.xaxis.set_pane_color(panels_color if x else c_transparent)
    ax.yaxis.set_pane_color(panels_color if y else c_transparent)
    ax.zaxis.set_pane_color(panels_color if z else c_transparent)

    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color


def decorator_set_pannels_3d(func, x=False, y=False, z=True, grey=0.95):

    def decorator(*args, **kwargs):

        func(*args, **kwargs)

        ax = args[0]

        set_pannels_3d(ax, x, y, z, grey)

    return decorator


def get_ax_3d(figsize=(4,4), constrained_layout=True, dpi=100):

    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout, dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    return ax


def get_ax_2d(figsize=(4, 4), constrained_layout=True, dpi=100):
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout, dpi=dpi)
    ax = fig.add_subplot()

    return ax


def get_ax_gridspec(rows, columns, size_ax=4, axs_3d=(), axs_ignored=()):

    fig = plt.figure(figsize=(columns*size_ax,rows*size_ax), constrained_layout=True, dpi=80)
    gs = fig.add_gridspec(ncols=columns,nrows=rows)

    axs = [[fig.add_subplot(gs[i,j], projection=('3d' if [i,j] in axs_3d else None)) for i in range(rows) if [i,j] not in axs_ignored] for j in range(columns)]

    return axs, gs, fig


def remove_ticks(ax):
    ax.set_xticks([], []), ax.set_yticks([], [])
    if ax.name == "3d": ax.set_zticks([], [])


def set_font(font_name='HelveticaNeue', font_size=default_font_size, font_color=(0, 0, 0), unicode_minus=True):

    font_path = ['/'.join(mdds.plotting.__file__.split('/')[:-1])]

    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.size'] = font_size
    matplotlib.rcParams["axes.unicode_minus"] = unicode_minus

    matplotlib.rcParams['text.color'] = font_color
    matplotlib.rcParams['axes.labelcolor'] = font_color
    matplotlib.rcParams['xtick.color'] = font_color
    matplotlib.rcParams['ytick.color'] = font_color


def set_centered_axes(ax, zero_centered=True, color=(0, 0, 0, 0.2), min_lim=0.1):

    if not zero_centered:
        y_max = max(np.max(np.abs(np.array(ax.get_ylim()))), min_lim)
        ax.set_ylim(-y_max, y_max)
        x_max = max(np.max(np.abs(np.array(ax.get_xlim()))), min_lim)
        ax.set_xlim(-x_max, x_max)
    else:
        '''ax.set_xlim(-min_lim if (-min_lim<ax.get_xlim()[0]<min_lim) else ax.get_xlim()[0],
                    min_lim if (-min_lim<ax.get_xlim()[1]<min_lim) else ax.get_xlim()[1])
        ax.set_ylim(-min_lim if (-min_lim<ax.get_ylim()[0]<min_lim) else ax.get_ylim()[0],
                    min_lim if (-min_lim<ax.get_ylim()[1]<min_lim) else ax.get_ylim()[1])'''
        ax.set_xlim(min(-min_lim, ax.get_xlim()[0]), max(min_lim, ax.get_xlim()[1]))
        ax.set_ylim(min(-min_lim, ax.get_ylim()[0]), max(min_lim, ax.get_ylim()[1]))

    if zero_centered:
        ax.spines['left'].set_position(('data',0))
        ax.spines['bottom'].set_position(('data',0))
    else:
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

    ax.spines['left'].set_linewidth(axes_line_width)
    ax.spines['bottom'].set_linewidth(axes_line_width)
    ax.spines['left'].set_capstyle('round')
    ax.spines['bottom'].set_capstyle('round')

    ax.tick_params(width=axes_line_width)

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color(color)
    ax.spines['bottom'].set_color(color)
    ax.spines['left'].set_zorder(1)
    ax.spines['bottom'].set_zorder(1)

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_zorder(1)
    ax.yaxis.set_zorder(1)

    #ax.ticklabel_format(axis='both', style='sci')
    """ax.get_xaxis().get_offset_text().set_position((-x_max, y_max/50))
    ax.get_yaxis().get_offset_text().set_position((x_max/50, y_max))"""


def set_bottom_axis(ax, color=(0, 0, 0)):
    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(axes_line_width)
        #ax.spines[side].set_color((0.2,0.2,0.2))
        ax.spines[side].set_capstyle('round')

    ax.spines['bottom'].set_color(color)
    ax.spines['right'].set_color((1, 1, 1, 0))
    ax.spines['top'].set_color((1, 1, 1, 0))
    ax.spines['left'].set_color((1, 1, 1, 0))
    ax.tick_params(width=axes_line_width, color=color)


def set_axes_linewidth(ax, color=None, linewidth=None):
    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(linewidth if linewidth is not None else axes_line_width)
        ax.spines[side].set_color(color if color is not None else (0.2, 0.2, 0.2))
        ax.spines[side].set_capstyle('round')

    ax.xaxis.set_tick_params(width=linewidth)
    ax.yaxis.set_tick_params(width=linewidth)


def set_axes_color_3d(ax, color):

    ax.w_xaxis.line.set_color(color)
    ax.w_yaxis.line.set_color(color)
    ax.w_zaxis.line.set_color(color)


def remove_axes(ax):
    for side in ax.spines.keys():
        ax.spines[side].set_color((1, 1, 1, 0))


def set_equal_lim(ax):

    x_max = np.max(np.abs(np.array(ax.get_xlim())))
    y_max = np.max(np.abs(np.array(ax.get_ylim())))
    if hasattr(ax, 'get_zlim'):
        z_max = np.max(np.abs(np.array(ax.get_zlim())))
    else:
        z_max = -1

    max_max = max(x_max, y_max, z_max)

    ax.set_xlim(-max_max, max_max)
    ax.set_ylim(-max_max, max_max)

    if hasattr(ax, 'get_zlim'):
        ax.set_zlim(-max_max, max_max)


def set_shared_lims(*axs):

    x_min, x_max = min([ax.get_xlim()[0] for ax in axs]), max([ax.get_xlim()[1] for ax in axs])
    for ax in axs: ax.set_xlim(x_min, x_max)

    y_min, y_max = min([ax.get_ylim()[0] for ax in axs]), max([ax.get_ylim()[1] for ax in axs])
    for ax in axs: ax.set_ylim(y_min, y_max)

    if hasattr(axs[0], 'get_zlim'):
        z_min, z_max = min([ax.get_zlim()[0] for ax in axs]), max([ax.get_zlim()[1] for ax in axs])
        for ax in axs: ax.set_zlim(z_min, z_max)


def set_centered_axes_3d(ax, c=(0.2,0.2,0.2), linewidth=1.5):

    ax.axis('off')

    ax_lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])

    ax.plot(ax_lims[0,:], [ax_lims[1,0],ax_lims[1,0]], [ax_lims[2,0],ax_lims[2,0]], color=c, linewidth=linewidth)
    ax.plot([ax_lims[0,0],ax_lims[0,0]], ax_lims[1,:], [ax_lims[2,0],ax_lims[2,0]], color=c, linewidth=linewidth)
    ax.plot([ax_lims[0,0], ax_lims[0,0]],[ax_lims[1,0],ax_lims[1,0]], ax_lims[2,:], color=c, linewidth=linewidth)

    ax.view_init(23, 45, roll=0)


def add_vertical_axes(ax, number_axes=3, spacing=0.15, color=(0.2, 0.2, 0.2)):

    axes = [ax.inset_axes([0,
                           (axi/number_axes)+(1/number_axes)*spacing,
                           1,
                           (1/number_axes)*(1-2*spacing)]) for axi in range(number_axes)]

    axes = axes.reverse()

    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for ax_i in axes: ax_i.set_facecolor(color)

    return axes
