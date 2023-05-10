from __future__ import annotations
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datar.apis.base import (
    all_,
    any_,
    any_na,
    append,
    outer,
    diff,
    duplicated,
    intersect,
    setdiff,
    setequal,
    unique,
    union,
    head,
    tail,
)

from .constants import NA
from ..utils import make_array, is_null, is_scalar, wrap_arrow_result
from ..arrow_ext import DatarArray


@all_.register(object, backend="arrow")
@wrap_arrow_result
def _all_(x):
    # If there is any NA, return NA
    if _any_na(x):
        return NA
    return pc.all(make_array(x).cast("bool"))


@any_.register(object, backend="arrow")
@wrap_arrow_result
def _any_(x) -> bool | pa.BooleanScalar:
    if isinstance(x, pa.BooleanScalar):
        return x.as_py() is True
    if is_scalar(x):
        return x is True
    return pc.any(make_array(x).cast("bool"))


@any_na.register(object, backend="arrow")
def _any_na(x) -> bool:
    return _any_(is_null(x))


@append.register(object, backend="arrow")
def _append(x, values, after: int = -1):
    x = make_array(x)
    if after is None:
        after = 0
    elif after < 0:
        after += len(x) + 1
    else:
        after += 1
    return make_array(np.insert(x, after, values), dtype=x.type)


@outer.register(object, backend="arrow")
def _outer(x, y, fun="*") -> List[pa.Array]:
    if fun == "*":
        return [make_array(o) for o in (np.outer(x, y))]

    kwargs = {}
    if getattr(fun, "_pipda_functype", None) in (
        "pipeable",
        "verb",
    ):  # pragma: no cover
        kwargs["__ast_fallback"] = "normal"
    return [
        make_array(fun(xi, make_array(y), **kwargs)) for xi in make_array(x)
    ]


@diff.register(object, backend="arrow")
def _diff(x, lag: int = 1, differences: int = 1):
    if lag != 1:
        raise ValueError("lag argument not supported")
    x = make_array(x)
    return make_array(np.diff(x, n=differences), dtype=x.type)


@duplicated.register(object, backend="arrow")
def _duplicated(x, incomparables=None, from_last: bool = False):
    dups = set()
    out = []
    out_append = out.append
    if incomparables is None:
        incomparables = []

    if from_last:
        x = reversed(x)
    for elem in x:
        if elem in incomparables:
            out_append(False)
        elif elem in dups:
            out_append(True)
        else:
            dups.add(elem)
            out_append(False)
    if from_last:
        out = list(reversed(out))
    return make_array(out, dtype="bool")


@intersect.register(object, backend="arrow")
@wrap_arrow_result
def _intersect(x, y):
    x = make_array(x)
    y = make_array(y)
    idx = pc.index_in(y.storage, x.storage).drop_null()
    return x.take(idx)


@setdiff.register(object, backend="arrow")
@wrap_arrow_result
def _setdiff(x, y):
    x = make_array(x)
    y = make_array(y)
    idx = pc.indices_nonzero(pc.invert(pc.is_in(x.storage, y.storage)))
    return x.storage.take(idx).unique()


@setequal.register(object, backend="arrow")
def _setequal(x, y):
    x = make_array(x).storage.unique().sort()
    y = make_array(y).storage.unique().sort()
    return x.equals(y)


@unique.register(object, backend="arrow")
@wrap_arrow_result
def _unique(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.unique(x)


@union.register(object, backend="arrow")
@wrap_arrow_result
def _union(x, y):
    x = make_array(x)
    y = make_array(y)
    out = pa.concat_arrays([x.storage, y.storage])
    return out.unique()


@head.register(object, backend="arrow")
def _head(x, n: int = 6):
    return make_array(x)[:n]


@tail.register(object, backend="arrow")
def _tail(x, n: int = 6):
    return make_array(x)[-n:]
