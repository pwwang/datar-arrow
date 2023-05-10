from __future__ import annotations
import math

import pyarrow.compute as pc
from datar.apis.base import (
    ceiling,
    cov,
    floor,
    mean,
    median,
    pmax,
    pmin,
    sqrt,
    var,
    scale,
    min_,
    max_,
    round_,
    sum_,
    abs_,
    prod,
    sign,
    signif,
    trunc,
    exp,
    log,
    log2,
    log10,
    log1p,
    sd,
    weighted_mean,
    quantile,
    proportions,
)
from ..utils import (
    is_scalar,
    make_array,
    transpose_arrays,
    wrap_arrow_result,
)


@ceiling.register(object, backend="arrow")
@wrap_arrow_result
def _ceiling(x):
    return pc.ceil(x)


@cov.register(object, backend="arrow")
@wrap_arrow_result
def _cov(x, y=None, na_rm: bool = False, ddof: int = 1):
    if y is None:
        raise ValueError(
            "In `cov(...)`: `y` must be provided if `x` is a vector"
        )
    x = make_array(x)
    y = make_array(y)
    n = len(x)
    if n != len(y):
        raise ValueError(
            "In `cov(...)`: `x` and `y` must have the same length"
        )

    xmean = pc.mean(x.storage, skip_nulls=na_rm)
    ymean = pc.mean(y.storage, skip_nulls=na_rm)
    x = x - xmean
    y = y - ymean
    return pc.divide(
        pc.sum(
            pc.multiply(x.storage, y.storage),
            skip_nulls=na_rm,
        ),
        (n - ddof),
    )


@floor.register(object, backend="arrow")
@wrap_arrow_result
def _floor(x):
    return pc.floor(x)


@mean.register(object, backend="arrow")
@wrap_arrow_result
def _mean(x, na_rm: bool = False):
    return pc.mean(x, skip_nulls=na_rm)


@median.register(object, backend="arrow")
@wrap_arrow_result
def _median(x, na_rm: bool = False):
    return pc.approximate_median(x, skip_nulls=na_rm)


@pmax.register(object, backend="arrow")
def _pmax(x, *more, na_rm: bool = False):
    arrs = transpose_arrays(x, *more)
    return make_array(
        pc.max(arr.storage, skip_nulls=na_rm).as_py() for arr in arrs
    )


@pmin.register(object, backend="arrow")
def _pmin(x, *more, na_rm: bool = False):
    arrs = transpose_arrays(x, *more)
    return make_array(
        pc.min(arr.storage, skip_nulls=na_rm).as_py() for arr in arrs
    )


@sqrt.register(object, backend="arrow")
@wrap_arrow_result
def _sqrt(x):
    return pc.sqrt(x)


@var.register(object, backend="arrow")
@wrap_arrow_result
def _var(x, na_rm: bool = False, ddof: int = 1):
    return pc.variance(x, ddof=ddof, skip_nulls=na_rm)


@scale.register(object, backend="arrow")
def _scale(x, center=True, scale_=True):
    center_true = center is True
    x = make_array(x)

    # center
    if center is True:
        center = pc.mean(x.storage)

    elif center is not False:
        center = make_array(center)

    if center is not False:
        x = x - center

    # scale
    if scale_ is True:
        if center_true:
            scale_ = pc.stddev(x.storage, ddof=1)
        else:
            scale_ = pc.sqrt(
                pc.divide(
                    pc.sum(pc.power(x.storage, 2)).cast("double"),
                    len(x) - 1,
                )
            )

    elif scale_ is not False:
        scale_ = make_array(scale_)

    if scale_ is not False:
        x = x / scale_

    return x


@min_.register(object, backend="arrow")
@wrap_arrow_result
def _min_(x, na_rm: bool = False):
    return pc.min(x, skip_nulls=na_rm)


@max_.register(object, backend="arrow")
@wrap_arrow_result
def _max_(x, na_rm: bool = False):
    return pc.max(x, skip_nulls=na_rm)


@round_.register(object, backend="arrow")
@wrap_arrow_result
def _round_(x, digits: int = 0):
    return pc.round(x, digits)


@sum_.register(object, backend="arrow")
@wrap_arrow_result
def _sum_(x, na_rm: bool = False):
    return pc.sum(x, skip_nulls=na_rm)


@abs_.register(object, backend="arrow")
@wrap_arrow_result
def _abs_(x):
    return pc.abs(x)


@prod.register(object, backend="arrow")
@wrap_arrow_result
def _prod(x, na_rm: bool = False):
    return pc.product(x, skip_nulls=na_rm)


@sign.register(object, backend="arrow")
@wrap_arrow_result
def _sign(x):
    return pc.sign(x)


@signif.register(object, backend="arrow")
def _signif(x, digits: int = 6):
    x = make_array(x)
    digits = pc.subtract(digits, pc.ceil(pc.log10(pc.abs(x.storage))))
    digits = _pmax(
        digits,
        0,
        na_rm=True,
    ).cast("int64")
    return make_array(round(i, d.as_py()) for i, d in zip(x, digits))


@trunc.register(object, backend="arrow")
@wrap_arrow_result
def _trunc(x):
    return pc.trunc(x)


@exp.register(object, backend="arrow")
@wrap_arrow_result
def _exp(x):
    return pc.exp(x)


@log.register(object, backend="arrow")
@wrap_arrow_result
def _log(x, base: float = math.e):
    return pc.divide(pc.log10(x), pc.log10(base))


@log2.register(object, backend="arrow")
@wrap_arrow_result
def _log2(x):
    return pc.log2(x)


@log10.register(object, backend="arrow")
@wrap_arrow_result
def _log10(x):
    return pc.log10(x)


@log1p.register(object, backend="arrow")
@wrap_arrow_result
def _log1p(x):
    return pc.log1p(x)


@sd.register(object, backend="arrow")
@wrap_arrow_result
def _sd(x, na_rm: bool = False, ddof: int = 1):
    return pc.stddev(x, ddof=ddof, skip_nulls=na_rm)


@weighted_mean.register(object, backend="arrow")
@wrap_arrow_result
def _weighted_mean(x, w=None, na_rm: bool = False):
    if w is None:
        return pc.mean(x, skip_nulls=na_rm)

    w = make_array(w, dtype="double")
    if w.null_count == len(w):
        return None

    if pc.equal(pc.sum(w.storage), 0).as_py():
        return None

    return pc.divide(
        pc.sum(
            pc.multiply(make_array(x).storage, w.storage), skip_nulls=na_rm
        ),
        pc.sum(w.storage, skip_nulls=na_rm).cast("double"),
    )


@quantile.register(object, backend="arrow")
@wrap_arrow_result
def _quantile(
    x,
    probs=(0.0, 0.25, 0.5, 0.75, 1.0),
    na_rm: bool = False,
    names: bool = True,  # not supported
    type_: int | str = 7,
    digits: int | str = 7,  # not supported
):
    methods = {
        # 1: "inverted_cdf",
        # 2: "averaged_inverted_cdf",
        # 3: "closest_observation",
        # 4: "interpolated_inverted_cdf",
        # 5: "hazen",
        # 6: "weibull",
        7: "linear",
        # 8: "median_unbiased",
        # 9: "normal_unbiased",
        10: "lower",
        11: "higher",
        12: "nearest",
        13: "midpoint",
    }
    # if numpy_version() < (1, 22):  # pragma: no cover
    #     kw = {"interpolation": methods.get(type_, type_)}
    # else:  # pragma: no cover
    #     kw = {"method": methods.get(type_, type_)}
    kw = {"interpolation": methods.get(type_, type_)}
    out = pc.quantile(x, probs, skip_nulls=na_rm, **kw)
    return out[0] if is_scalar(probs) else out


@proportions.register(object, backend="arrow")
@wrap_arrow_result
def _proportions(x, margin=None):
    x = make_array(x)
    return pc.divide(x.storage.cast("float64"), pc.sum(x.storage))
