"""Plotting utilities."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations

from geometric_bayesian.utils.types import Callable


def simple_plot(
    fn: Callable | jax.Array,
    range=[-1, 1],
    res=100,
    fig=None,
    **kwargs
):
    if fig is not None:
        ax = fig.get_axes()[0]
    else:
        fig, ax = plt.subplots()

    if isinstance(fn, Callable):
        x = jnp.linspace(range[0], range[1], res)
        ax.plot(x, fn(x))
    else:
        ax.plot(*fn)

    ax.set_xlim(range[0], range[1])

    fig.tight_layout()
    return fig


def scatter_plot(data, range=None, fig=None, **kwargs):
    if fig is not None:
        ax = fig.get_axes()[0]
    else:
        if len(data) <= 2:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data, **kwargs)
    if range is not None:
        ax.set_xlim(range[0], range[1])
    return fig


def contour_plot(
        fn,
        grid=None,
        min=[-1, -1],
        max=[1, 1],
        res=100,
        xlabel=None,
        ylabel=None,
        iso=False,
        cbar=False,
        fig=None,
        **kwargs
):
    if fig is not None:
        ax = fig.get_axes()
    else:
        fig, ax = plt.subplots()

    if grid is None:
        x, y, z = _get_xyz(fn, min, max, res)
    else:
        z = fn
        x, y = grid

    im = ax.contourf(x, y, z, 500, cmap="Spectral")
    if iso:
        ax.contour(x, y, z, 10, cmap=None, colors='#f2e68f')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([x.min(), x.max()])
    ax.set_yticks([y.min(), y.max()])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)
    ax.axis('equal')

    if cbar:
        _colorbar(im, fig, ax, **kwargs)

    # fig.patch.set_visible(False)
    fig.tight_layout()
    return fig


def contour_nd_plot(
        fn,
        ranges,
        res=50,
        reduce_fn=jnp.sum,
        iso=False,
        cbar=False,
        axes_label=None,
        **kwargs
):
    dim = len(ranges)
    grids = [jnp.linspace(start, end, res) for (start, end) in ranges]
    mesh = jnp.meshgrid(*grids, indexing='ij')
    # F = fn(*mesh)
    F = fn(jnp.stack([g.ravel() for g in mesh], axis=-1)).reshape(mesh[0].shape)

    pairs = list(combinations(range(dim), 2))
    n_plots = len(pairs)
    ncols = int(jnp.ceil(jnp.sqrt(n_plots)))
    nrows = int(jnp.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols)
    axs = [item for row in axs for item in row]
    for idx, (i, j) in enumerate(pairs):
        ax = axs[idx]
        reduce_axes = tuple(k for k in range(dim) if k not in (i, j))
        F_proj = reduce_fn(F, axis=reduce_axes)

        Xi, Xj = jnp.meshgrid(grids[i], grids[j], indexing='ij')
        im = ax.contourf(Xi, Xj, F_proj, levels=500, cmap="Spectral")
        if iso:
            ax.contour(Xi, Xj, F_proj, 10, cmap=None, colors='#f2e68f')

        # ax.set_title(f"Projection onto axes ({i}, {j})")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([Xi.min(), Xi.max()])
        ax.set_yticks([Xj.min(), Xj.max()])
        ax.set_xlabel(f"${axes_label}_1$", labelpad=-5)
        ax.set_ylabel(f"${axes_label}_2$", labelpad=-20)
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        ax.axis('equal')

        if cbar:
            _colorbar(im, fig, ax, **kwargs)

    for ax in axs[n_plots:]:
        ax.axis('off')

    # fig.patch.set_visible(False)
    fig.tight_layout()
    return fig


def surf_plot(fn, min=[-1, -1], max=[1, 1], res=100, fig_ax=None, **kwargs):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    x, y, z = get_xyz(fn, min, max, res)
    im = ax.plot_surface(x, y, z, cmap="Spectral", antialiased=True, alpha=0.8,)
    ax.set_box_aspect((jnp.ptp(x), jnp.ptp(y), jnp.ptp(z)))
    ax.axis('off')
    # ax.view_init(elev=30, azim=-95)

    fig.patch.set_visible(False)
    fig.tight_layout()

    return fig


def _colorbar(im, fig, ax, pos="left", size="8%", pad=0.4, label=None, labelpad=-50, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
    cax.yaxis.set_ticks_position(pos)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0.0])
    if label is not None:
        cbar.set_label('$' + label + '$', fontsize=18, rotation=90, labelpad=labelpad)


def _get_xyz(fn, min, max, res):
    if isinstance(fn, Callable):
        x, y = jnp.meshgrid(jnp.linspace(min[0], max[0], res), jnp.linspace(min[1], max[1], res))
        eval_fn = fn(jnp.vstack((x.ravel(), y.ravel())).transpose())
        if len(eval_fn.shape) == 1:
            z = eval_fn.reshape(res, res)
        else:
            x, y, z = eval_fn[:, 0].reshape(res, res), eval_fn[:, 1].reshape(res, res), eval_fn[:, 2].reshape(res, res)
    else:
        res = int(jnp.sqrt(fn.shape[0]))
        x, y, z = fn[:, 0].reshape(res, res), fn[:, 1].reshape(res, res), fn[:, 2].reshape(res, res)
    return x, y, z


def plot_regression(model, X_test, y_test=None, X_train=None, y_train=None, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    pred = model(X_test)
    ax.plot(X_test, pred, label='map')
    if y_test is not None:
        ax.plot(X_test, y_test, label='target')
    if X_train is not None and y_train is not None:
        ax.scatter(X_train, y_train, label='samples', color='green', alpha=0.5)
    ax.legend()

    return fig
