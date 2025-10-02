"""Plotting utilities."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations

from geometric_bayesian.utils.types import Callable


def colorbar(im, fig, ax, pos="left", size="8%", pad=0.4, label=None, labelpad=-50, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
    cax.yaxis.set_ticks_position(pos)
    cbar.outline.set_visible(False)
    # cbar.set_ticks([])
    if label is not None:
        cbar.set_label('$' + label + '$', fontsize=18, rotation=90, labelpad=labelpad)


def get_xyz(fn, min, max, res):
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


def contour_plot(fn, min=[-1, -1], max=[1, 1], res=100, iso=True, alpha=1.0, zorder=0, cbar=True, fig_ax=None, **kwargs):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    x, y, z = get_xyz(fn, min, max, res)
    im = ax.contourf(x, y, z, 500, cmap="Spectral", alpha=alpha, zorder=zorder)
    if iso:
        ax.contour(x, y, z, 10, cmap=None, colors='#f2e68f', zorder=zorder)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axis('off')
    ax.axis('equal')

    if cbar:
        colorbar(im, fig, ax, **kwargs)
    fig.patch.set_visible(False)
    fig.tight_layout()

    return fig


def contour_nd_plot(fn, ranges, resolution=50, reduce_fn=jnp.sum, iso=False, alpha=1.0, zorder=0, cbar=False, **kwargs):
    dim = len(ranges)
    grids = [jnp.linspace(start, end, resolution) for (start, end) in ranges]
    mesh = jnp.meshgrid(*grids, indexing='ij')
    # F = fn(*mesh)
    F = fn(jnp.stack([g.ravel() for g in mesh], axis=-1)).reshape(mesh[0].shape)

    pairs = list(combinations(range(dim), 2))
    n_plots = len(pairs)
    ncols = int(jnp.ceil(jnp.sqrt(n_plots)))
    nrows = int(jnp.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols)
    axes = [item for row in axes for item in row]

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        reduce_axes = tuple(k for k in range(dim) if k not in (i, j))
        F_proj = reduce_fn(F, axis=reduce_axes)
        xi = grids[i]
        xj = grids[j]
        Xi, Xj = jnp.meshgrid(xi, xj, indexing='ij')

        im = ax.contourf(Xi, Xj, F_proj, levels=500, cmap="Spectral", alpha=alpha, zorder=zorder)
        if iso:
            ax.contour(Xi, Xj, F_proj, 10, cmap=None, colors='#f2e68f', zorder=zorder)

        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        ax.axis('equal')
        # ax.axis('off')
        ax.set_title(f"Projection onto axes ({i}, {j})")
        ax.set_xlabel(f"x{i}", labelpad=-5)
        ax.set_ylabel(f"x{j}", labelpad=-20)
        ax.set_xticks([Xi.min(), Xi.max()])
        ax.set_yticks([Xj.min(), Xj.max()])
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        if cbar:
            # fig.colorbar(im, ax=ax)
            colorbar(im, fig, ax, **kwargs)

    for ax in axes[n_plots:]:
        ax.axis('off')
    fig.patch.set_visible(False)
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
