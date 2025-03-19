"""Plotting utilities."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_regression_with_uncertainty(
    train_input,
    train_target,
    X_grid,
    Y_pred,
    Y_var,
    title="Model Predictions with Uncertainty",
    xlabel="Input",
    ylabel="Target",
):
    """Plot training data, model predictions, and uncertainty.

    Args:
        train_input: Training inputs (e.g., X_train).
        train_target: Training targets (e.g., y_train).
        X_grid: Input points for predictions.
        Y_pred: Predicted mean values.
        Y_var: Predicted variance (for uncertainty bounds).
        title: Plot title (default: "Model Predictions with Uncertainty").
        xlabel: Label for the x-axis (default: "Input").
        ylabel: Label for the y-axis (default: "Target").
    """
    _fig, ax = plt.subplots(figsize=(8, 6))

    # Plot training points
    ax.scatter(
        train_input,
        train_target,
        label="Training Points",
        color="green",
        s=50,
        edgecolor="k",
    )

    # Plot predicted mean
    ax.plot(X_grid, Y_pred, label="Prediction Mean", color="blue", linewidth=2)

    # Add uncertainty band
    ax.fill_between(
        X_grid[:, 0],
        Y_pred - 1.96 * Y_var,
        Y_pred + 1.96 * Y_var,
        color="cornflowerblue",
        alpha=0.3,
        label="Confidence Interval (95%)",
    )

    # Customize plot appearance
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)

    # Show the plot
    plt.show()


def create_reliability_diagram(
    bin_confidences: jax.Array,
    bin_accuracies: jax.Array,
    num_bins: int,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(visible=True, axis="y")

    bar_centers = jnp.linspace(0, 1, num_bins + 1)[:-1] + 1 / (2 * num_bins)
    bar_width = 1 / num_bins

    ax.bar(
        x=bar_centers,
        height=bin_accuracies,
        width=bar_width,
        label="Outputs",
        color="blue",
        edgecolor="black",
    )

    ax.bar(
        x=bar_centers,
        height=bin_confidences - bin_accuracies,
        width=bar_width / 2,
        bottom=bin_accuracies,
        label="Gap",
        color="red",
        edgecolor="red",
        alpha=0.4,
    )

    ax.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle="--", color="gray")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    fig.legend()

    ax.set_aspect("equal")

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()

    else:
        plt.show()


def create_proportion_diagram(
    bin_proportions: jax.Array,
    num_bins: int,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(visible=True, axis="y")

    bar_centers = jnp.linspace(0, 1, num_bins + 1)[:-1] + 1 / (2 * num_bins)
    bar_width = 1 / num_bins

    ax.bar(
        x=bar_centers,
        height=bin_proportions,
        width=bar_width,
        label="Proportions",
        color="green",
        edgecolor="black",
        alpha=0.4,
    )

    ax.axhline(y=1 / num_bins, color="gray", linestyle="--", label="Uniform")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Proportion")
    fig.legend()

    ax.set_aspect("equal")

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()

    else:
        plt.show()


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
    x, y = jnp.meshgrid(jnp.linspace(min[0], max[0], res), jnp.linspace(min[1], max[1], res))
    eval_fn = fn(jnp.vstack((x.ravel(), y.ravel())).transpose()).reshape(res, res)

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    im = ax.contourf(x, y, eval_fn, 500, cmap="viridis", alpha=alpha, zorder=zorder)
    if iso:
        ax.contour(x, y, eval_fn, 10, cmap=None, colors='#f2e68f', zorder=zorder)
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
    eval_fn = fn(jnp.vstack((x.ravel(), y.ravel())).transpose()).reshape(res, res)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    else:
        ax = fig.axes[0]
    im = ax.plot_surface(x, y, eval_fn, cmap="viridis", antialiased=True, alpha=0.8,)
    ax.axis('off')
    ax.view_init(elev=30, azim=-95)
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
