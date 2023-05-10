import numpy as np
from datar.apis.base import (
    beta,
    lgamma,
    digamma,
    trigamma,
    choose,
    factorial,
    gamma,
    lfactorial,
    lchoose,
    lbeta,
    psigamma,
)

from .bessel import _get_special_func_from_scipy
from ..utils import is_scalar, make_array


@beta.register(object, backend="arrow")
def _beta(x, y):
    out = make_array(_get_special_func_from_scipy("beta")(x, y))
    return out[0] if is_scalar(x) and is_scalar(y) else out


@lgamma.register(object, backend="arrow")
def _lgamma(x):
    out = make_array(_get_special_func_from_scipy("gammaln")(x))
    return out[0] if is_scalar(x) else out


@digamma.register(object, backend="arrow")
def _digamma(x):
    out = make_array(_get_special_func_from_scipy("psi")(x))
    return out[0] if is_scalar(x) else out


@trigamma.register(object, backend="arrow")
def _trigamma(x):
    z = _get_special_func_from_scipy("polygamma")(1, x)

    print(z, type(z), np.isnan(z))
    out = make_array(_get_special_func_from_scipy("polygamma")(1, x))
    return out[0] if is_scalar(x) else out


@choose.register(object, backend="arrow")
def _choose(n, k):
    out = make_array(_get_special_func_from_scipy("binom")(n, k))
    return out[0] if is_scalar(n) and is_scalar(k) else out


@factorial.register(object, backend="arrow")
def _factorial(x):
    out = make_array(_get_special_func_from_scipy("factorial")(x))
    return out[0] if is_scalar(x) else out


@gamma.register(object, backend="arrow")
def _gamma(x):
    out = make_array(_get_special_func_from_scipy("gamma")(x))
    return out[0] if is_scalar(x) else out


@lfactorial.register(object, backend="arrow")
def _lfactorial(x):
    out = np.log(make_array(_get_special_func_from_scipy("factorial")(x)))
    return out[0] if is_scalar(x) else out


@lchoose.register(object, backend="arrow")
def _lchoose(n, k):
    out = np.log(make_array(_get_special_func_from_scipy("binom")(n, k)))
    return out[0] if is_scalar(n) and is_scalar(k) else out


@lbeta.register(object, backend="arrow")
def _lbeta(x, y):
    out = make_array(_get_special_func_from_scipy("betaln")(x, y))
    return out[0] if is_scalar(x) and is_scalar(y) else out


@psigamma.register(object, backend="arrow")
def _psigamma(x, deriv):
    out = make_array(
        _get_special_func_from_scipy("polygamma")(np.round(deriv), x)
    )
    return out[0] if is_scalar(x) else out
