# %%
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as normal
from jax.scipy.stats.multivariate_normal import pdf as multivariate_normal
jax.config.update('jax_enable_x64', True)

# import matplotlib
# matplotlib.use('module://matplotlib-backend-sixel')

# %% prior distribution
std_theta_1 = jnp.sqrt(10.0)
std_theta_2 = jnp.sqrt(1.0)
prior_mean = jnp.zeros(2)
prior_cov = jnp.array([[std_theta_1, 0.0], [0.0, std_theta_2]])
p_theta = lambda theta: multivariate_normal(theta, mean=prior_mean, cov=prior_cov)


theta_x, theta_y = jnp.meshgrid(jnp.linspace(-3, 3, 100), jnp.linspace(-3, 3, 100))
theta_grid = jnp.vstack((theta_x.ravel(), theta_y.ravel())).T
fig, ax = plt.subplots()
ax.contour(theta_x, theta_y, p_theta(theta_grid).reshape(100, -1))
fig.show()

# %% conditional
theta_mode = jnp.array([0.0, 0.0])
std_x = np.sqrt(2.0)
p_x_theta = lambda x, theta: 0.5 * normal(x, loc=theta[0], scale=std_x) + 0.5 * normal(x, loc=theta.sum(), scale=std_x)
fig, ax = plt.subplots()
x_grid = jnp.linspace(-3, 3, 100)
ax.plot(x_grid, p_x_theta(x_grid, theta_mode))
fig.show()

# %% sample from likelihood
n_samples = 1000
np.random.seed(42)
np.random.binomial(1, 0.5, size=n_samples)
mixture_choices = np.random.binomial(1, 0.5, size=n_samples)
x_samples = np.where(
    mixture_choices == 0,
    np.random.normal(theta_mode[0], std_x, size=n_samples),
    np.random.normal(theta_mode.sum(), std_x, size=n_samples)
)

# likelihood = jax.vmap(p_x_theta, in_axes=(0, 0), out_axes=(0, 1))
likelihood = lambda theta: jnp.sum(jax.vmap(lambda x: p_x_theta(x, theta))(x_samples))
fig, ax = plt.subplots()
ax.contour(theta_x, theta_y, jnp.exp(jax.vmap(likelihood)(theta_grid).reshape(100, -1)))
fig.show()
