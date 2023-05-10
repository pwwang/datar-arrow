from __future__ import annotations
from typing import Any, Callable

import pyarrow as pa
import pyarrow.compute as pc

from .utils import wrap_arrow_result, get_dtype, wrap_arrow_value


def _floor_divide(x: pa.Array, y: pa.Array):
    return pc.floor(pc.divide(x, y)).cast("int64")


def _mod(x: pa.Array, y: pa.Array):
    return pc.subtract(x, pc.multiply(_floor_divide(x, y), y))


def _binop(
    fn: Callable,
    x: Any,
    y: Any,
    wrap: bool = True,
):
    """Binary operation"""
    if isinstance(x, DatarArray):
        x = x.storage
    if isinstance(y, DatarArray):
        y = y.storage

    if isinstance(x, pa.Array) and isinstance(y, pa.Array):
        if len(x) == 1 and len(y) > 1:
            x = x[0]
        if len(y) == 1 and len(x) > 1:
            y = y[0]

    out = fn(x, y)
    return wrap_arrow_value(out) if wrap else out


class DatarArray(pa.ExtensionArray):
    """Extend pyarrow.Array to support arithmetic operators

    Unless pyarrow supports them natively, we will implement them here.
    """

    dictionary = None
    indices = None
    _dictionary_array = None

    def __add__(self, other):
        return _binop(pc.add, self, other)

    def __radd__(self, other):
        return _binop(pc.add, other, self)

    def __sub__(self, other):
        return _binop(pc.subtract, self, other)

    def __rsub__(self, other):
        return _binop(pc.subtract, other, self)

    def __mul__(self, other):
        return _binop(pc.multiply, self, other)

    def __rmul__(self, other):
        return _binop(pc.multiply, other, self)

    def __truediv__(self, other):
        return _binop(pc.divide, self, other)

    def __rtruediv__(self, other):
        return _binop(pc.divide, other, self)

    def __pow__(self, other):
        return _binop(pc.power, self, other)

    def __rpow__(self, other):
        return _binop(pc.power, other, self)

    def __eq__(self, other):
        return _binop(pc.equal, self, other)

    def __ne__(self, other):
        return _binop(pc.not_equal, self, other)

    def __lt__(self, other):
        return _binop(pc.less, self, other)

    def __le__(self, other):
        return _binop(pc.less_equal, self, other)

    def __gt__(self, other):
        return _binop(pc.greater, self, other)

    def __ge__(self, other):
        return _binop(pc.greater_equal, self, other)

    @wrap_arrow_result
    def __neg__(self):
        return pc.negate(self.storage)

    @wrap_arrow_result
    def __abs__(self):
        return pc.abs(self.storage)

    @wrap_arrow_result
    def __invert__(self):
        return pc.invert(self.storage)

    def __and__(self, other):
        return _binop(pc.bit_wise_and, self, other)

    def __rand__(self, other):
        return _binop(pc.bit_wise_and, other, self)

    def __or__(self, other):
        return _binop(pc.bit_wise_or, self, other)

    def __ror__(self, other):
        return _binop(pc.bit_wise_or, other, self)

    def __xor__(self, other):
        return _binop(pc.bit_wise_xor, self, other)

    def __rxor__(self, other):
        return _binop(pc.bit_wise_xor, other, self)

    def __lshift__(self, other):
        return _binop(pc.shift_left, self, other)

    def __rlshift__(self, other):
        return _binop(pc.shift_left, other, self)

    def __rshift__(self, other):
        return _binop(pc.shift_right, self, other)

    def __rrshift__(self, other):
        return _binop(pc.shift_right, other, self)

    def __floordiv__(self, other):
        return _binop(_floor_divide, self, other)

    def __rfloordiv__(self, other):
        return _binop(_floor_divide, other, self)

    @wrap_arrow_result
    def __getitem__(self, idx):
        try:
            return self.storage[idx]
        except TypeError:
            # list, np.ndarray, etc
            return self.take(idx)

    @wrap_arrow_result
    def __mod__(self, other):
        return _binop(_mod, self, other)

    @wrap_arrow_result
    def __rmod__(self, other):
        return _binop(_mod, other, self)

    def __pos__(self):
        return self

    def __array__(self, dtype=None):
        return self.storage.__array__(dtype)

    def __iter__(self):
        return iter(self.storage.to_pylist())

    @wrap_arrow_result
    def take(self, indices, **kwargs):
        if isinstance(indices, DatarArray):
            indices = indices.storage
        return self.storage.take(indices, **kwargs)

    @property
    def type(self):
        return self.storage.type

    @classmethod
    def create(cls, arr):
        if isinstance(arr, pa.DictionaryArray):
            values = arr.dictionary_decode()
            out = cls.from_storage(DatarArrayType(values.type), values)
            out.dictionary = arr.dictionary
            out.indices = arr.indices
            out._dictionary_array = arr
            return out

        return pa.ExtensionArray.from_storage(DatarArrayType(arr.type), arr)


class DatarArrayType(pa.PyExtensionType):
    def __init__(self, t: type | pa.DataType | str):
        pa.PyExtensionType.__init__(self, get_dtype(t))

    def __reduce__(self):
        return DatarArrayType, ()

    def __arrow_ext_class__(self):
        return DatarArray
