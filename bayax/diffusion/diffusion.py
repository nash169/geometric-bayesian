import jax
import jax.numpy as jnp


def diffusion(f, dt, **kwargs):
    def fn(t, x, u, key):
        drift, cov = f(t, x, u)
        cov_sqrtf = cov.sqrtf(**kwargs)
        diff = jnp.sqrt(dt) * (cov_sqrtf @ jax.random.normal(key, (cov_sqrtf.shape[1],), dtype=x.dtype))
        return drift, diff
    return fn
