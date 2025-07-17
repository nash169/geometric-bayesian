
import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve
from jax.lax import while_loop
from jax.random import normal


def apply_perturbation_if_needed(alpha, beta, eps):
    """
    Apply a perturbation to eigenvalues and off-diagonals if eigenvalue gaps are too small.
    """
    n = len(alpha)

    # Compute gaps between consecutive eigenvalues
    gaps = jnp.abs(alpha[1:] - alpha[:-1])

    # Define a gap threshold (TOL * |W(I)|, where TOL = 10 * EPS)
    tol = 10 * eps
    gap_threshold = tol * jnp.abs(alpha[:-1])

    # Identify indices where gaps are too small
    small_gap_indices = jnp.where(gaps < gap_threshold, size=len(gaps), fill_value=-1)[0]

    # Apply a small perturbation at these indices
    perturbation = eps * jnp.arange(1, n)
    perturbed_alpha = alpha
    perturbed_alpha = perturbed_alpha.at[small_gap_indices + 1].add(perturbation[small_gap_indices])
    perturbed_beta = beta + eps * jnp.arange(1, n)

    return perturbed_alpha, perturbed_beta


def eigvec_tridiagonal(key, alpha, beta, eigvals):
    """Implements inverse iteration to compute eigenvectors in JAX."""
    k = eigvals.shape[0]
    n = alpha.shape[0]

    eps = jnp.finfo(eigvals.dtype).eps
    alpha, beta = apply_perturbation_if_needed(alpha, beta, eps)

    t_norm = jnp.maximum(jnp.abs(eigvals[0]), jnp.abs(eigvals[-1]))
    gaptol = jnp.sqrt(eps) * t_norm

    # Identify clusters of close eigenvalues
    gap = eigvals[1:] - eigvals[:-1]
    close = gap < gaptol
    left_neighbor_close = jnp.concatenate([jnp.array([False]), close])
    right_neighbor_close = jnp.concatenate([close, jnp.array([False])])

    max_clusters = n  # Maximum possible clusters
    ortho_interval_start = jnp.where(~left_neighbor_close & right_neighbor_close, size=max_clusters, fill_value=-1)[0]
    ortho_interval_end = jnp.where(left_neighbor_close & ~right_neighbor_close, size=max_clusters, fill_value=-1)[0] + 1
    # num_clusters = jnp.sum(ortho_interval_start != -1)

    # Initialize random starting vectors
    v0 = normal(key, (k, n), dtype=alpha.dtype)
    v0 = v0 / jnp.linalg.norm(v0, axis=1, keepdims=True)

    alpha_shifted = alpha[None, :] - eigvals[:, None]
    beta_tiled = jnp.tile(beta[None, :], (k, 1))

    # Pad beta_tiled to create dl and du with the required shape
    dl = jnp.pad(beta_tiled, [(0, 0), (1, 0)])  # Add leading zero
    du = jnp.pad(beta_tiled, [(0, 0), (0, 1)])  # Add trailing zero

    def orthogonalize_cluster(vectors, start, end):
        # Create a mask for the range [start:end]
        indices = jnp.arange(vectors.shape[0])
        mask = (indices >= start) & (indices < end)

        # Apply the mask to extract the cluster
        cluster = vectors * mask[:, None]  # Mask along the rows
        cluster = jnp.where(mask[:, None], cluster, 0)  # Ensure masked rows are zero

        # QR decomposition on the selected rows
        q, _ = jnp.linalg.qr(cluster.T)

        # Align q to the appropriate rows of the original array
        aligned_q = jnp.where(mask[:, None], q.T, 0)

        # Update the original vectors using jnp.where
        updated_vectors = jnp.where(mask[:, None], aligned_q, vectors)
        return updated_vectors

    def orthogonalize_close_eigenvectors(v):
        def body(i, v):
            start = ortho_interval_start[i]
            end = ortho_interval_end[i]
            return jax.lax.cond(
                (start != -1) & (end != -1),
                lambda args: orthogonalize_cluster(args[0], args[1], args[2]),
                lambda args: args[0],
                (v, start, end)
            )

        return jax.lax.fori_loop(0, max_clusters, body, v)

    def continue_iteration(state):
        i, _, nrm_v, nrm_v_old = state
        max_it = 5
        min_norm_growth = 0.1
        norm_growth_factor = 1 + min_norm_growth
        return jnp.logical_and(
            i < max_it, jnp.any(nrm_v >= norm_growth_factor * nrm_v_old)
        )

    def iteration_step(state):
        i, v, nrm_v, nrm_v_old = state

        # Use vmap to handle batched tridiagonal solve
        def solve_tridiagonal_single(diags, rhs):
            return tridiagonal_solve(diags[0], diags[1], diags[2], rhs[:, None])[:, 0]

        dl_batched = dl  # Lower diagonal
        d_batched = alpha_shifted  # Middle diagonal
        du_batched = du  # Upper diagonal

        # Stack diagonals to match the batch structure
        batched_diags = (dl_batched, d_batched, du_batched)

        # Apply batched tridiagonal solve
        v = jax.vmap(solve_tridiagonal_single, in_axes=(0, 0))(batched_diags, v.T).T

        nrm_v_old = nrm_v
        nrm_v = jnp.linalg.norm(v, axis=0)
        v = v / nrm_v[None, :]

        v = orthogonalize_close_eigenvectors(v)
        return i + 1, v, nrm_v, nrm_v_old

    nrm_v = jnp.linalg.norm(v0, axis=1)
    zero_nrm = jnp.zeros_like(nrm_v)
    state = (0, v0, nrm_v, zero_nrm)

    _, v, _, _ = while_loop(continue_iteration, iteration_step, state)
    return v
