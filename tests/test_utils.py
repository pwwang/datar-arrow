import pytest  # noqa
import pyarrow as pa
from datar.core import plugin  # noqa
from datar.base import NA
from datar_arrow.utils import (
    broadcast_arrays,
    is_scalar,
    make_array,
    get_dtype,
    wrap_arrow_value,
)
from .utils import assert_iterable_equal


def test_is_scalar():
    assert is_scalar(1)
    assert is_scalar(pa.lib.Int64Array)


def test_make_array():
    assert_iterable_equal(make_array(1), [1])
    assert_iterable_equal(make_array(pa.scalar(1)), [1])
    assert_iterable_equal(make_array([1, 2]), [1, 2])
    assert_iterable_equal(make_array(pa.array([1, 2])), [1, 2])
    # assert_iterable_equal(make_array({"a": 1, "b": 2}), ["a", "b"])
    # assert_iterable_equal(make_array(["1", "2"], dtype=int), [1, 2])
    assert_iterable_equal(make_array(["1", NA]), ["1", NA])


@pytest.mark.parametrize(
    "intype,outtype",
    [
        ("int8", pa.int8()),
        ("int16", pa.int16()),
        ("int32", pa.int32()),
        ("int64", pa.int64()),
        ("uint8", pa.uint8()),
        ("uint16", pa.uint16()),
        ("uint32", pa.uint32()),
        ("uint64", pa.uint64()),
        ("float32", pa.float32()),
        ("float64", pa.float64()),
        ("bool_", pa.bool_()),
        ("str", pa.string()),
        (pa.int8(), pa.int8()),
        (None, None),
        ("int", pa.int64()),
        (int, pa.int64()),
    ],
)
def test_get_dtype(intype, outtype):
    assert get_dtype(intype) is outtype


def test_get_dtype_error():
    with pytest.raises(TypeError):
        get_dtype(object())


def test_wrap_arrow_value():
    x = make_array(1)
    assert wrap_arrow_value(x) is x

    x = pa.array([1, 2])
    assert wrap_arrow_value(x).storage.to_pylist() == x.to_pylist()

    x = pa.scalar(1)
    assert wrap_arrow_value(x) == 1

    x = object()
    assert wrap_arrow_value(x) is x


def test_broadcast_arrays_error():
    with pytest.raises(ValueError, match="Arrays must be"):
        broadcast_arrays([1, 2], [1, 2, 3])
