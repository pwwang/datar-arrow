from __future__ import annotations

import inspect
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:  # pragma: no cover
    from .arrow_ext import DatarArray

DTYPE_MAP = {
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "str": pa.string(),
    "datetime": pa.timestamp("ns"),
    "date": pa.date32(),
    "time": pa.time32("s"),
    "timedelta": pa.duration("ns"),
    "bytes": pa.binary(),
    "double": pa.float64(),
}


def get_dtype(x: type | pa.DataType | str | None) -> pa.DataType | None:
    """Get the pyarrow type from a type or pyarrow type"""
    if x is None:
        return None

    if isinstance(x, pa.DataType):
        return x

    if isinstance(x, str) and hasattr(pa, x):
        return getattr(pa, x)()

    if isinstance(x, str) and x in DTYPE_MAP:
        return DTYPE_MAP[x]

    try:
        return DTYPE_MAP.get(x.__name__, x)
    except AttributeError:
        raise TypeError(f"Invalid type: {x}") from None


def wrap_arrow_value(x: Any) -> Any:
    """Ensure x is not raw pyarrow type

    Since pyarrow types do not support arithmetic operators, we need to
    convert them to DatarArray or other supported types.

    Args:
        x: The object to check

    Returns:
        x if x is not a pyarrow type, otherwise converted value
        If x is pyarrow.Array, it will be converted to DatarArray.
        If x is pyarrow.Scalar, it will be converted to a python scalar.
        Otherwise, keep x as is.
    """
    from .arrow_ext import DatarArray

    if isinstance(x, DatarArray):
        # If x is already a DatarArray, return it directly
        return x

    if isinstance(x, pa.Array):
        return DatarArray.create(x)

    if isinstance(x, pa.Scalar):
        return x.as_py()

    return x


def wrap_arrow_result(fn: Callable) -> Callable:
    """Decorator to ensure the return value is not a pyarrow type"""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return wrap_arrow_value(fn(*args, **kwargs))

    return wrapper


def is_scalar(x: Any) -> bool:
    """Is x a scalar?
    Using np.ndim() instead of np.isscalar() to allow an object (i.e. a dict)
    a scalar, because it can be used as data in a cell of a dataframe.

    Args:
        x: The object to check

    Returns:
        True if x is a scalar, False otherwise
    """
    if isinstance(x, (list, set, tuple, dict)):
        # np.ndim({'a'}) == 0
        return False

    if isinstance(x, type):
        return True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
        return np.ndim(x) == 0


@wrap_arrow_result
def is_null(x: Any) -> bool | pa.BooleanArray:
    """Is x None or NA? Like pandas.isnull()

    Args:
        x: The object to check

    Returns:
        If x is scalar, return True if x is None or NA, False otherwise.
        If x is an array, return a boolean array with the same shape as x.
    """
    from .arrow_ext import DatarArray

    if isinstance(x, DatarArray):
        x = x.storage
    return pc.is_null(x, nan_is_null=True)


def make_array(x: Any, dtype: type | pa.DataType = None) -> DatarArray:
    """Make an array from x"""
    from .arrow_ext import DatarArray

    if isinstance(x, DatarArray):
        return x

    dtype = get_dtype(dtype)
    if isinstance(x, np.ndarray):
        # convert nan to null
        if np.ndim(x) == 0:
            x = x.ravel()
        x = pa.array(
            x,
            type=dtype,
            mask=pc.is_null(x, nan_is_null=True).to_numpy(
                zero_copy_only=False
            ),
        )
    elif isinstance(x, pa.Array):
        x = x.cast(dtype) if dtype is not None else x
    elif inspect.isgenerator(x):
        x = pa.array(x, type=dtype)
    elif isinstance(x, pa.Scalar):
        x = pa.array([x.as_py()], type=dtype)
    elif is_scalar(x):
        x = pa.array([x], type=dtype)
    else:
        x = pa.array(x, type=dtype)

    return DatarArray.create(x)


def flatten_slice(x: slice) -> DatarArray[int]:
    """Flatten a slice into an array of integers"""
    start = x.start or 0
    stop = x.stop or 0
    if x.step == 1:
        stop += 1
    step = 1 if x.step is None else x.step
    return make_array(range(start, stop, step))


def broadcast_arrays(*arrs: Any) -> Tuple[pa.Array]:
    """Broadcast arrays to the same shape"""
    arrs = tuple(make_array(arr) for arr in arrs)
    lens = set(len(arr) for arr in arrs)
    maxlen = max(lens)

    if len(lens) == 1:
        return arrs

    if len(lens) > 2 or 1 not in lens:
        raise ValueError("Arrays must be of length 1 or the max length")

    return tuple(
        arr if len(arr) == maxlen else make_array(pa.repeat(arr[0], maxlen))
        for arr in arrs
    )


def transpose_arrays(*arrs: Any) -> Tuple[pa.Array]:
    """Transpose arrays"""
    arrs = broadcast_arrays(*arrs)
    # Is there a better way to do this?
    # Without as_py(), the following error occurs:
    # did not recognize Python value type when inferring an Arrow data type
    return tuple(make_array((a for a in arr)) for arr in zip(*arrs))
