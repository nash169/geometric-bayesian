from geometric_bayesian.densities import MultivariateNormal
from geometric_bayesian.operators import ScaledOperator


def proposal(f, t, x, u, dt):
    drift, cov = f(t=t, x=x, u=u)
    return MultivariateNormal(cov=ScaledOperator(dt, cov), mean=x + dt * drift)
