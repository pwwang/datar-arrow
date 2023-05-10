import pytest  # noqa: F401
import numpy as np
import pyarrow as pa
from datar_arrow.arrow_ext import DatarArray
from .utils import assert_equal, assert_iterable_equal


def test_binop():
    x = pa.array([1, 2, 3])
    x = DatarArray.create(x)

    assert_iterable_equal(x + 1, [2, 3, 4])
    assert_iterable_equal(1 + x, [2, 3, 4])
    assert_iterable_equal(x + pa.array([1]), [2, 3, 4])
    assert_iterable_equal(pa.array([1]) + x, [2, 3, 4])

    assert_iterable_equal(x - 1, [0, 1, 2])
    assert_iterable_equal(1 - x, [0, -1, -2])

    assert_iterable_equal(x * 2, [2, 4, 6])
    assert_iterable_equal(2 * x, [2, 4, 6])

    assert_iterable_equal(x / 2., [0.5, 1, 1.5])
    assert_iterable_equal(2. / x, [2., 1., 2. / 3.], approx=True)

    assert_iterable_equal(x // 2, [0, 1, 1])
    assert_iterable_equal(2 // x, [2, 1, 0])

    assert_iterable_equal(x ** 2, [1, 4, 9])
    assert_iterable_equal(2 ** x, [2, 4, 8])

    assert_iterable_equal(x % 2, [1, 0, 1])
    assert_iterable_equal(2 % x, [0, 0, 2])

    assert_iterable_equal(x & 2, [0, 2, 2])
    assert_iterable_equal(2 & x, [0, 2, 2])

    assert_iterable_equal(x | 2, [3, 2, 3])
    assert_iterable_equal(2 | x, [3, 2, 3])

    assert_iterable_equal(x ^ 2, [3, 0, 1])
    assert_iterable_equal(2 ^ x, [3, 0, 1])

    assert_iterable_equal(x << 2, [4, 8, 12])
    assert_iterable_equal(2 << x, [4, 8, 16])

    assert_iterable_equal(x >> 2, [0, 0, 0])
    assert_iterable_equal(2 >> x, [1, 0, 0])

    assert_iterable_equal(x == 2, [False, True, False])
    assert_iterable_equal(x != 2, [True, False, True])
    assert_iterable_equal(x < 2, [True, False, False])
    assert_iterable_equal(x <= 2, [True, True, False])
    assert_iterable_equal(x > 2, [False, False, True])
    assert_iterable_equal(x >= 2, [False, True, True])


def test_unaryop():
    x = pa.array([1, 2, 3])
    x = DatarArray.create(x)

    assert_iterable_equal(+x, [1, 2, 3])
    assert_iterable_equal(-x, [-1, -2, -3])
    assert_iterable_equal(abs(-x), [1, 2, 3])
    assert_iterable_equal(abs(x), [1, 2, 3])

    x = pa.array([True, False, True])
    x = DatarArray.create(x)
    assert_iterable_equal(~x, [False, True, False])


def test_getitem():
    x = pa.array([1, 2, 3])
    x = DatarArray.create(x)

    assert_equal(x[0], 1)
    assert_equal(x[1], 2)
    assert_equal(x[2], 3)
    assert_iterable_equal(x[[0, 1]], [1, 2])
    assert_iterable_equal(x[:2], [1, 2])


def test_np_array():
    x = pa.array([1, 2, 3])
    x = DatarArray.create(x)

    assert_iterable_equal(np.tile(x, 2), [1, 2, 3, 1, 2, 3])


def test_take():
    x = pa.array([1, 2, 3])
    x = DatarArray.create(x)

    assert_iterable_equal(x.take([0, 1]), [1, 2])
    assert_iterable_equal(x.take(x[:2]), [2, 3])


def test_type():
    x = pa.array([1, 2, 3])
    x = DatarArray.create(x)
    assert x.type == pa.int64()
