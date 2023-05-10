import pytest
import pyarrow as pa
from datar.base import (
    all_,
    any_,
    any_na,
    append,
    diff,
    outer,
    duplicated,
    intersect,
    setdiff,
    setequal,
    unique,
    union,
    head,
    tail,
    NA,
)

from datar_arrow.utils import make_array
from .utils import assert_iterable_equal, assert_equal


def test_all_():
    assert_equal(all_([True, True, True]), True)
    assert_equal(all_([True, True, False]), False)
    assert_equal(all_([True, True, NA]), NA)


def test_any_():
    assert_equal(any_([True, True, True]), True)
    assert_equal(any_([True, True, False]), True)
    assert_equal(any_([False, False, False]), False)
    assert_equal(any_(pa.scalar(True)), True)
    assert_equal(any_(True), True)


def test_any_na():
    assert_equal(any_na([True, True, True]), False)
    assert_equal(any_na([True, True, False]), False)
    assert_equal(any_na([False, False, False]), False)
    assert_equal(any_na([False, False, NA]), True)
    assert_equal(any_na([NA, NA, NA]), True)


def test_append():
    assert_iterable_equal(append([1, 2, 3], 4), [1, 2, 3, 4])
    assert_iterable_equal(append([1, 2, 3], 4, after=0), [1, 4, 2, 3])
    assert_iterable_equal(append([1, 2, 3], 4, after=None), [4, 1, 2, 3])
    assert_iterable_equal(append([1, 2, 3], 4, after=1), [1, 2, 4, 3])
    assert_iterable_equal(append([1, 2, 3], 4, after=-1), [1, 2, 3, 4])
    assert_iterable_equal(append([1, 2, 3], 4, after=-2), [1, 2, 4, 3])


def test_diff():
    assert_iterable_equal(diff([1, 2, 3, 4, 5]), [1, 1, 1, 1])
    assert_iterable_equal(diff([1, 2, 3, 4, 5], differences=2), [0, 0, 0])
    with pytest.raises(ValueError):
        diff([1, 2, 3, 4, 5], lag=2)


def test_duplicated():
    assert_iterable_equal(
        duplicated([1, 2, 3, 4, 5]), [False, False, False, False, False]
    )
    assert_iterable_equal(
        duplicated([1, 2, 3, 4, 5, 1]),
        [False, False, False, False, False, True],
    )
    assert_iterable_equal(
        duplicated([1, 2, 3, 4, 5, 1], from_last=True),
        [True, False, False, False, False, False],
    )
    assert_iterable_equal(
        duplicated([1, 2, 3, 4, 5, 1], incomparables=[1]),
        [False, False, False, False, False, False],
    )


def test_intersect():
    assert_iterable_equal(intersect([1, 2, 3], [3, 4, 5]), [3])


def test_setdiff():
    assert_iterable_equal(setdiff([1, 2, 3], [3, 4, 5]), [1, 2])


def test_setequal():
    assert_equal(setequal([1, 2, 3], [3, 4, 5]), False)
    assert_equal(setequal([1, 2, 3], [3, 2, 1]), True)


def test_unique():
    assert_iterable_equal(unique([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
    assert_iterable_equal(unique([1, 2, 3, 4, 5, 1]), [1, 2, 3, 4, 5])
    assert_iterable_equal(unique(make_array([1, 2, 3, 4, 1])), [1, 2, 3, 4])


def test_union():
    assert_iterable_equal(union([1, 2, 3], [3, 4, 5]), [1, 2, 3, 4, 5])


def test_head():
    assert_iterable_equal(head([1, 2, 3, 4, 5, 6, 7]), [1, 2, 3, 4, 5, 6])
    assert_iterable_equal(head([1, 2, 3, 4, 5], 2), [1, 2])


def test_tail():
    assert_iterable_equal(tail([1, 2, 3, 4, 5, 6, 7]), [2, 3, 4, 5, 6, 7])
    assert_iterable_equal(tail([1, 2, 3, 4, 5], 2), [4, 5])


def test_outer():
    out = outer([1, 2], [1, 2, 3])
    assert len(out) == 2
    assert_iterable_equal(out[0], [1, 2, 3])
    assert_iterable_equal(out[1], [2, 4, 6])

    out = outer([1, 2], [1, 2, 3], fun=lambda a, b: a + b)
    assert len(out) == 2
    assert_iterable_equal(out[0], [2, 3, 4])
    assert_iterable_equal(out[1], [3, 4, 5])
