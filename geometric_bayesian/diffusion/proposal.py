from bayax.densities import MultivariateNormal
from bayax.operators import ScaledOperator


def proposal(f, t, x, u, dt):
    drift, cov = f(t=t, x=x, u=u)
    return MultivariateNormal(cov=ScaledOperator(dt, cov), mean=x + dt * drift)
