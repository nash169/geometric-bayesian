"""Plotting utilities."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from geometric_bayesian.utils.types import Callable


def colorbar(im, fig, ax, pos="left", size="5%", pad=0.2, label=None, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=ticks)
    cax.yaxis.set_ticks_position(pos)
    cbar.outline.set_visible(False)
    cbar.set_ticks([])
    if label is not None:
        cbar.set_label('$'+label+'$', fontsize=24, rotation=0, labelpad=(-50 if pos == "left" else +50))


def contour_plot(fn, min=[-1, -1], max=[1, 1], res=100, iso=True, alpha=1.0, zorder=0, fig=None):
    if isinstance(fn, Callable):
        x, y = jnp.meshgrid(jnp.linspace(min[0], max[0], res), jnp.linspace(min[1], max[1], res))
        z = fn(jnp.vstack((x.ravel(), y.ravel())).transpose()).reshape(res, res)
    else:
        x, y, z = fn

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    im = ax.contourf(x, y, z, 500, cmap="viridis", alpha=alpha, zorder=zorder)
    if iso:
        ax.contour(x, y, z, 10, cmap=None, colors='#f2e68f', zorder=zorder)
    colorbar(im, fig, ax, size='8%')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('equal')
    fig.tight_layout()

    return fig


def surf_plot(fn, min=[-1, -1], max=[1, 1], res=100, fig=None):
    x, y = jnp.meshgrid(jnp.linspace(min[0], max[0], res), jnp.linspace(min[1], max[1], res))
    eval_fn = fn(jnp.vstack((x.ravel(), y.ravel())).transpose())
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    else:
        ax = fig.axes[0]
    im = ax.plot_surface(eval_fn[:, 0].reshape(res, res), eval_fn[:, 1].reshape(res, res),
                         eval_fn[:, 2].reshape(res, res), cmap="Spectral", antialiased=True, alpha=0.8,)
    ax.set_box_aspect((jnp.ptp(eval_fn[:, 0]), jnp.ptp(eval_fn[:, 1]), jnp.ptp(eval_fn[:, 2])))
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
