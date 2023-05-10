from __future__ import annotations

import numpy as np
import pyarrow as pa
from datar.core.utils import logger
from datar.apis.base import (
    rep,
    c_,
    length,
    lengths,
    order,
    sort,
    rank,
    rev,
    sample,
    seq,
    seq_along,
    seq_len,
    match,
)
from ..utils import is_null, make_array, is_scalar, wrap_arrow_result


@rep.register(object, backend="arrow")
def _rep(
    x,
    times=1,
    length=None,
    each=1,
):
    x = make_array(x)
    xtype = x.type
    times = make_array(times)
    length = make_array(length)
    each = make_array(each)
    if len(times) == 1:
        times = times[0]
    if len(length) >= 1:
        if len(length) > 1:
            logger.warning(
                "[datar_arrow] "
                "In rep(...): first element used of 'length' argument"
            )
        length = length[0]
    if len(each) == 1:
        each = each[0]

    if not is_scalar(times):
        if len(times) != len(x):
            raise ValueError(
                "Invalid times argument, expect length "
                f"{len(x)}, got {len(times)}"
            )

        if not isinstance(each, int) or each != 1:
            raise ValueError(
                "Unexpected each argument when times is an iterable."
            )

    if is_scalar(times) and isinstance(times, int):
        x = make_array(np.tile(np.repeat(x, each), times), dtype=xtype)
    else:
        x = make_array(np.repeat(x, times), dtype=xtype)

    if length is None:
        return x

    repeats = length // len(x) + 1
    x = np.tile(x, repeats)

    return make_array(x[:length], dtype=xtype)


@c_.register(object, backend="arrow")
@wrap_arrow_result
def _c(*args):
    return pa.concat_arrays(
        [
            make_array(xi).storage
            if is_scalar(xi)
            else c_(*xi, __backend="arrow", __ast_fallback="normal").storage
            for xi in args
        ]
    )


@length.register(object, backend="arrow")
def _length(x):
    return len(make_array(x))


@lengths.register(object, backend="arrow")
def _lengths(x) -> pa.IntegerArray:
    return (
        make_array(1)
        if is_scalar(x)
        else make_array([len(make_array(xi)) for xi in x])
    )


@order.register(object, backend="arrow")
def _order(x, decreasing: bool = False, na_last: bool = True):
    and_ = not na_last and decreasing
    or_ = not na_last or decreasing
    na = -np.inf if or_ and not and_ else np.inf

    x = make_array(x)
    xtype = x.type
    x = np.where(is_null(x), na, x)
    out = np.argsort(x)
    return make_array(out[::-1] if decreasing else out, dtype=xtype)


@sort.register(object, backend="arrow")
def _sort(x, decreasing: bool = False, na_last: bool = True):
    x = make_array(x)
    idx = order(
        x,
        decreasing=decreasing,
        na_last=na_last,
        __ast_fallback="normal",
        __backend="arrow",
    )
    return x.take(idx)


@rank.register(object, backend="arrow")
def _rank(x, na_last: bool = True, ties_method: str = "average"):
    if not na_last:
        raise NotImplementedError("na_last=False is not supported yet")

    try:
        from scipy import stats
    except ImportError as imperr:  # pragma: no cover
        raise ImportError(
            "`rank` requires `scipy` package.\n" "Try: pip install -U scipy"
        ) from imperr

    return make_array(stats.rankdata(x, method=ties_method))


@rev.register(object, backend="arrow")
def _rev(x):
    return make_array(x)[::-1]


@sample.register(object, backend="arrow")
def _sample(
    x,
    size: int = None,
    replace: bool = False,
    prob: float | np.ndarray[float] = None,
):
    x = make_array(x)
    size = len(x) if size is None else int(size)

    return make_array(
        np.random.choice(x, size, replace=replace, p=prob),
        dtype=x.type,
    )


@seq.register(object, backend="arrow")
def _seq(
    from_,
    to=None,
    by=None,
    length_out=None,
    along_with=None,
):
    if along_with is not None:
        return seq_along(
            along_with, __backend="arrow", __ast_fallback="normal"
        )

    if not is_scalar(from_):
        return seq_along(from_, __backend="arrow", __ast_fallback="normal")

    if length_out is not None and from_ is None and to is None:
        return seq_len(length_out, __backend="arrow", __ast_fallback="normal")

    if from_ is None:
        from_ = 1
    elif to is None:
        from_, to = 1, from_

    if length_out is not None:
        by = (float(to) - float(from_)) / float(length_out)
    elif by is None:
        by = 1 if to > from_ else -1
        length_out = to - from_ + 1 if to > from_ else from_ - to + 1
    else:
        length_out = (to - from_ + 1.1 * by) // by

    return make_array([from_ + n * by for n in range(int(length_out))])


@seq_along.register(object, backend="arrow")
def _seq_along(x):
    return make_array(np.arange(len(make_array(x))) + 1)


@seq_len.register((list, tuple, pa.Array), backend="arrow")
def _seq_len_obj(length_out):
    if len(length_out) > 1:
        logger.warning(
            "[datar_arrow] In seq_len(...): "
            "first element used of 'length_out' argument"
        )

    length_out = length_out[0]
    if isinstance(length_out, pa.Scalar):
        length_out = length_out.as_py()
    return make_array(np.arange(length_out) + 1)


@seq_len.register(
    (
        int,
        np.integer,
        pa.Int8Scalar,
        pa.Int16Scalar,
        pa.Int32Scalar,
        pa.Int64Scalar,
    ),
    backend="arrow",
)
def _seq_len_int(length_out):
    length_out = (
        length_out.as_py() if isinstance(length_out, pa.Scalar) else length_out
    )
    return make_array(np.arange(length_out) + 1)


@match.register(object, backend="arrow")
def _match(x, table, nomatch=-1):
    sorter = np.argsort(table)
    searched = np.searchsorted(table, x, sorter=sorter).ravel()
    out = sorter.take(searched, mode="clip")
    out[~np.isin(x, table)] = nomatch
    return make_array(out)
