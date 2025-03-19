from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
from jax import random


class DataLoader:
    """Simple dataloader."""

    def __init__(self, X, y, batch_size, *, shuffle=True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = X.shape[0]
        self.indices = np.arange(self.dataset_size)
        self.rng = np.random.default_rng(seed=0)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        return self.X[batch_indices], self.y[batch_indices]


# Function to create the sinusoid dataset
def get_sinusoid_example(
    num_data: int = 150,
    sigma_noise: float = 0.3,
    batch_size: int = 150,
    rng_key=None,
) -> tuple[
    jnp.ndarray, jnp.ndarray, Iterator[tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray, jnp.ndarray
]:
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    """Generate a sinusoid dataset.

    Args:
        num_data: Number of data points.
        sigma_noise: Standard deviation of the noise.
        batch_size: Batch size for the data loader.
        rng_key: Random number generator key.

    Returns:
        X_train: Training input data.
        y_train: Training target data.
        train_loader: Data loader for training data.
        X_test: Testing input data.
    """
    # Split RNG key for reproducibility
    rng_key, rng_noise = random.split(rng_key)

    # Generate random training data
    X_train = (
        random.uniform(rng_key, (num_data, 1)) * 10
    )  # X_train values between 0 and 8
    noise = random.normal(rng_noise, X_train.shape) * sigma_noise
    y_train = jnp.sin(X_train) + noise

    # Generate testing data
    X_test = jnp.linspace(-5, 15, 500).reshape(-1, 1)
    y_test = jnp.sin(X_test)

    # Create the training data loader
    train_loader = DataLoader(X_train, y_train, batch_size)

    return X_train, y_train, train_loader, X_test, y_test
