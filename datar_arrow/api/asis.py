from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as patypes
from datar.apis.base import (
    is_atomic,
    is_character,
    is_complex,
    is_double,
    is_integer,
    is_element,
    is_finite,
    is_false,
    is_infinite,
    is_logical,
    is_na,
    is_null,
    is_numeric,
    is_true,
    as_character,
    as_complex,
    # as_date,
    as_double,
    as_integer,
    as_logical,
    as_null,
    as_numeric,
)

from ..utils import is_scalar, make_array, wrap_arrow_result
from ..arrow_ext import DatarArray
from .constants import NULL


@is_atomic.register(object, backend="arrow")
def _is_atomic(x: Any) -> bool:
    return is_scalar(x)


@is_character.register(object, backend="arrow")
def _is_character(x: Any) -> bool:
    t = make_array(x).type
    return patypes.is_string(t) or patypes.is_large_string(t)


@is_complex.register(object, backend="arrow")
def _is_complex(x: Any) -> bool:  # pragma: no cover
    raise NotImplementedError("is_complex() is not supported by arrow backend")


@is_double.register(object, backend="arrow")
def _is_double(x: Any) -> bool:
    return patypes.is_float64(make_array(x).type)


@is_integer.register(object, backend="arrow")
def _is_integer(x: Any) -> bool:
    return patypes.is_integer(make_array(x).type)


@is_element.register(object, backend="arrow")
@wrap_arrow_result
def _is_element(x: Any, y: Any) -> bool:
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.is_in(x, make_array(y).storage)


@is_finite.register(object, backend="arrow")
@wrap_arrow_result
def _is_finite(x: Any) -> bool | pa.BooleanArray:
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.is_finite(x)


@is_false.register(object, backend="arrow")
def _is_false(x: Any) -> bool:
    if isinstance(x, pa.BooleanScalar):
        x = x.as_py()
    return x is False


@is_infinite.register(object, backend="arrow")
@wrap_arrow_result
def _is_infinite(x: Any) -> bool | pa.BooleanArray:
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.is_inf(x)


@is_logical.register(object, backend="arrow")
def _is_logical(x: Any) -> bool:
    return patypes.is_boolean(make_array(x).type)


@is_na.register(object, backend="arrow")
@wrap_arrow_result
def _is_na(x: Any) -> bool | pa.BooleanArray:
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.is_nan(x)


@is_null.register(object, backend="arrow")
def _is_null(x: Any) -> bool:
    return x is NULL or isinstance(x, pa.NullScalar)


@is_numeric.register(object, backend="arrow")
def _is_numeric(x: Any) -> bool:
    dtype = make_array(x).type
    return patypes.is_integer(dtype) or patypes.is_floating(dtype)


@is_true.register(object, backend="arrow")
def _is_true(x: Any) -> bool:
    if isinstance(x, pa.BooleanScalar):
        x = x.as_py()
    return x is True


@as_character.register(object, backend="arrow")
@wrap_arrow_result
def _as_character(x: Any) -> str | pa.BooleanArray:
    x_scalar = is_scalar(x)
    out = make_array(x).cast("string")
    return out[0] if x_scalar else out


@as_complex.register(object, backend="arrow")
def _as_complex(x: Any):  # pragma: no cover
    raise NotImplementedError("as_complex() is not supported by arrow backend")


@as_double.register(object, backend="arrow")
@wrap_arrow_result
def _as_double(x: Any) -> float | pa.FloatingPointArray:
    x_scalar = is_scalar(x)
    out = make_array(x).cast("double")
    return out[0] if x_scalar else out


@as_integer.register(object, backend="arrow")
@wrap_arrow_result
def _as_integer(x: Any) -> int | pa.IntegerArray:
    x_scalar = is_scalar(x)
    out = make_array(x)
    if patypes.is_floating(out.type):
        out = pc.floor(out.storage).cast("int64")
    else:
        out = out.cast("int64")
    return out[0] if x_scalar else out


@as_integer.register(pa.DictionaryArray, backend="arrow")
@wrap_arrow_result
def _as_integer_dict_array(x: pa.DictionaryArray) -> int | pa.IntegerArray:
    return x.indices


@as_integer.register(DatarArray, backend="arrow")
@wrap_arrow_result
def _as_integer_factor(x: DatarArray) -> int | pa.IntegerArray:
    if x.dictionary is None:
        return _as_integer(x)
    return x.indices


@as_logical.register(object, backend="arrow")
@wrap_arrow_result
def _as_logical(x: Any) -> bool | pa.BooleanArray:
    x_scalar = is_scalar(x)
    out = make_array(x)
    if patypes.is_string(out.type):
        out = pc.invert(pc.match_like(out.storage, ""))
    else:
        out = out.cast("bool")
    return out[0] if x_scalar else out


@as_null.register(object, backend="arrow")
def _as_null(x: Any) -> None:
    return NULL


@as_numeric.register(object, backend="arrow")
@wrap_arrow_result
def _as_numeric(x: Any) -> Number | np.ndarray[Number]:
    x_scalar = is_scalar(x)
    x = make_array(x)
    try:
        out = x.cast("int64")
    except pa.ArrowInvalid:
        try:
            out = x.cast("double")
        except pa.ArrowInvalid:
            raise ValueError(f"Cannot convert {x} to numeric") from None

    return out[0] if x_scalar else out
