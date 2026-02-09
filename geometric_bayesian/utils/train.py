import tqdm
import optax
import jax.numpy as jnp
from flax import nnx


def train(
    model,
    dataloader,
    loss_fn,
    params: dict,
    opt=optax.adam,
    key=None,
    verbose: bool = False,
    record: bool = False
):
    assert "lr" in params, "define lerning rate [lr]"
    assert "n_iters" in params, "define number of iterations [n_iters]"

    optimizer = nnx.Optimizer(model, opt(params["lr"]), wrt=nnx.Param)
    iters = tqdm.tqdm(range(params["n_iters"]), desc="Epoch", disable=not verbose)

    if key is None:
        @nnx.jit
        def step(model, optimizer, x, y):
            loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(
                model, x, y
            )
            optimizer.update(model, grads)
            return loss
    else:
        @nnx.jit
        def step(model, optimizer, x, y, key):
            (loss, key), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(
                model, x, y, key
            )
            optimizer.update(model, grads)
            return loss, key

    loss_log, weights_log = [], []
    for i in iters:
        for x_tr, y_tr in dataloader:
            if key is None:
                loss = step(model, optimizer, x_tr, y_tr)
            else:
                loss, key = step(model, optimizer, x_tr, y_tr, key)
            if verbose:
                iters.set_postfix({"Loss": loss.item()})
            if record:
                loss_log.append(loss)
                weights_log.append(model.params)

    return jnp.array(loss_log), jnp.array(weights_log)
