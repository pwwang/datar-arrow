import pytest  # noqa
import pyarrow as pa
from datar.base import (
    droplevels,
    factor,
    levels,
    as_factor,
    is_factor,
    nlevels,
    is_ordered,
    ordered,
    NA,
)
from datar_arrow.utils import make_array
from .utils import assert_iterable_equal, assert_equal


def test_droplevels():
    fct = factor([1, 2, 3], levels=[1, 2, 3, 4])
    out = droplevels(fct)
    assert_iterable_equal(levels(out), [1, 2, 3])

    fct = pa.DictionaryArray.from_arrays(fct.indices, fct.dictionary)
    out = droplevels(fct)
    assert_iterable_equal(levels(out), [1, 2, 3])

    with pytest.raises(NotImplementedError):
        droplevels(make_array(1), __ast_fallback="normal")


def test_levels():
    lvls = levels(1)
    assert lvls is None


def test_factor():
    out = factor()
    assert_equal(len(out), 0)
    assert_equal(len(levels(out)), 0)
    assert_equal(len(factor(2)), 1)

    out = factor([1, 2, 3], exclude=None)
    assert_equal(len(out), 3)

    out = factor([1, 2, 3], exclude=1)
    assert_iterable_equal(out, [NA, 2, 3])
    assert_iterable_equal(levels(out), [2, 3])

    out = factor(out)
    assert_iterable_equal(out, [NA, 2, 3])
    assert_iterable_equal(levels(out), [2, 3])

    with pytest.raises(NotImplementedError):
        factor([1, 2, 3], ordered=True, __ast_fallback="normal")


def test_as_factor():
    out = as_factor([1, 2, 3])
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])

    out2 = as_factor(out)
    assert out is out2

    out3 = as_factor(out._dictionary_array)
    assert_iterable_equal(out3, [1, 2, 3])
    assert_iterable_equal(levels(out3), [1, 2, 3])

    out4 = as_factor(make_array([1, 2, 3]))
    assert_iterable_equal(out4, [1, 2, 3])
    assert_iterable_equal(levels(out4), [1, 2, 3])


def test_is_factor():
    out = as_factor([])
    isf1 = is_factor(out)
    assert isf1
    isf2 = is_factor([])
    assert not isf2
    isf3 = is_factor(out._dictionary_array)
    assert isf3


def test_nlevels():
    assert_equal(nlevels(1), 0)
    assert_equal(nlevels(factor([1, 2, 3])), 3)


def test_is_ordered():
    iso1 = is_ordered(1)
    assert not iso1
    iso2 = is_ordered(factor())
    assert not iso2


def test_ordered():
    with pytest.raises(NotImplementedError):
        ordered([1, 2, 3], __ast_fallback="normal")
