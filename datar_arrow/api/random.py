import builtins
import random as _random

import numpy as np
from datar.apis.base import (
    set_seed,
    rnorm,
    runif,
    rpois,
    rbinom,
    rcauchy,
    rchisq,
    rexp,
)
from ..utils import is_scalar, make_array


@set_seed.register(object, backend="arrow")
def _set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)


@rnorm.register(object, backend="arrow")
def _rnorm(n, mean=0, sd=1):
    n = n if is_scalar(n) else max(n)
    return make_array(np.random.normal(mean, sd, n))


@runif.register(object, backend="arrow")
def _runif(n, min=0, max=1):
    n = n if is_scalar(n) else builtins.max(n)
    return make_array(np.random.uniform(min, max, n))


@rpois.register(object, backend="arrow")
def _rpois(n, lambda_):
    n = n if is_scalar(n) else max(n)
    return make_array(np.random.poisson(lambda_, n))


@rbinom.register(object, backend="arrow")
def _rbinom(n, size, prob):
    n = n if is_scalar(n) else max(n)
    return make_array(np.random.binomial(size, prob, n))


@rcauchy.register(object, backend="arrow")
def _rcauchy(n, location=0, scale=1):
    n = n if is_scalar(n) else max(n)
    # Don't involve the Index from Series in arithmetics
    # which is error-prone
    scale = getattr(scale, "values", scale)
    location = getattr(location, "values", location)
    return make_array(np.random.standard_cauchy(n) * scale + location)


@rchisq.register(object, backend="arrow")
def _rchisq(n, df):
    n = n if is_scalar(n) else max(n)
    return make_array(np.random.chisquare(df, n))


@rexp.register(object, backend="arrow")
def _rexp(n, rate=1):
    n = n if is_scalar(n) else max(n)
    return make_array(np.random.exponential(1 / rate, n))
