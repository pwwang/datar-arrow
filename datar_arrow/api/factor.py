"""Implement factor using pyarrow's dictionary array"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from datar.apis.base import (
    factor,
    ordered,
    levels,
    nlevels,
    droplevels,
    is_factor,
    as_factor,
    is_ordered,
)
from ..utils import make_array, wrap_arrow_result
from ..arrow_ext import DatarArray


@factor.register(object, backend="arrow")
@wrap_arrow_result
def _factor(
    x=None,
    *,
    levels=None,
    labels=None,
    exclude=None,
    ordered=False,
    nmax=None,
) -> pa.DictionaryArray:
    if ordered:
        raise NotImplementedError("ordered factor is not supported yet")

    if x is None:
        return make_array([]).storage.dictionary_encode()

    if levels is None and exclude is None:
        return make_array(x).storage.dictionary_encode()

    x = make_array(x).storage
    if levels is None:
        levels = x.unique().drop_null()
    else:
        levels = make_array(levels).storage

    if exclude is not None:
        exclude = make_array(exclude).storage
        levels = levels.filter(pc.invert(pc.is_in(levels, exclude)))

    indices = pc.index_in(x, levels)
    return pa.DictionaryArray.from_arrays(indices, levels)


@ordered.register(object, backend="arrow")
def _ordered(x):
    raise NotImplementedError("ordered factor is not supported yet")


@levels.register(object, backend="arrow")
def _levels(x) -> pa.Array | None:
    return None


@levels.register((pa.DictionaryArray, DatarArray), backend="arrow")
@wrap_arrow_result
def _levels_dictionary_array(x) -> pa.Array:
    return x.dictionary


@nlevels.register(object, backend="arrow")
def _nlevels(x) -> int:
    return 0


@nlevels.register((pa.DictionaryArray, DatarArray), backend="arrow")
def _nlevels_dictionary_array(x) -> int:
    return 0 if x.dictionary is None else len(x.dictionary)


@droplevels.register(pa.DictionaryArray, backend="arrow")
def _droplevels_dictionary_array(x: pa.DictionaryArray) -> DatarArray:
    values = x.dictionary.take(x.indices)
    lvls = x.dictionary.filter(
        pc.is_in(x.dictionary, values.unique().drop_null())
    )
    return _factor(values, levels=lvls)


@droplevels.register(DatarArray, backend="arrow")
def _droplevels_datar_array(x: DatarArray) -> DatarArray:
    if x.dictionary is None:
        raise NotImplementedError("droplevels on non-factor is not supported")

    values = x.storage
    lvls = x.dictionary.filter(
        pc.is_in(x.dictionary, values.unique().drop_null())
    )
    return _factor(values, levels=lvls)


@is_factor.register(object, backend="arrow")
def _is_factor(x) -> bool:
    if not isinstance(x, pa.Array):
        return False
    if isinstance(x, DatarArray):
        return x.dictionary is not None
    return pa.types.is_dictionary(x.type)


@is_ordered.register(object, backend="arrow")
def _is_ordered(x) -> bool:
    return False


@as_factor.register(object, backend="arrow")
def _as_factor(x) -> pa.DictionaryArray:
    return _factor(x)


@as_factor.register(pa.DictionaryArray, backend="arrow")
@wrap_arrow_result
def _as_factor_dictionary_array(x) -> pa.DictionaryArray:
    return x


@as_factor.register(DatarArray, backend="arrow")
def _as_factor_datar_array(x) -> pa.DictionaryArray:
    if x.dictionary is not None:
        return x

    return _factor(x.storage)
