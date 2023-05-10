import pytest  # noqa: F401
import pyarrow.compute as pc
from datar.base import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    cospi,
    sin,
    sinh,
    sinpi,
    tan,
    tanh,
    tanpi,
    atan2,
    pi,
)
from datar_arrow.utils import make_array
from .utils import assert_equal, assert_iterable_equal


def test_acos():
    assert_equal(acos(0.5), pc.acos(0.5).as_py(), approx=True)
    assert_iterable_equal(
        acos(make_array([0.5, 0.6])),
        [pc.acos(0.5).as_py(), pc.acos(0.6).as_py()],
        approx=True,
    )


def test_acosh():
    assert_equal(acosh(1.5), 0.9624236501192069, approx=True)
    assert_iterable_equal(
        acosh(make_array([1.5, 1.6])), [0.9624236501192069, 1.0469679150031885],
        approx=True,
    )


def test_asin():
    assert_equal(asin(0.5), pc.asin(0.5).as_py(), approx=True)
    assert_iterable_equal(
        asin(make_array([0.5, 0.6])),
        [pc.asin(0.5).as_py(), pc.asin(0.6).as_py()],
        approx=True,
    )


def test_asinh():
    assert_equal(asinh(1.5), 1.1947632172871094, approx=True)
    assert_iterable_equal(
        asinh(make_array([1.5, 1.6])), [1.1947632172871094, 1.2489833279048763],
        approx=True,
    )


def test_atan():
    assert_equal(atan(0.5), pc.atan(0.5).as_py(), approx=True)
    assert_iterable_equal(
        atan(make_array([0.5, 0.6])),
        [pc.atan(0.5).as_py(), pc.atan(0.6).as_py()],
        approx=True,
    )


def test_atanh():
    assert_equal(atanh(0.5), 0.5493061443340548, approx=True)
    assert_iterable_equal(
        atanh(make_array([0.5, 0.6])),
        [0.5493061443340548, 0.6931471805599453],
        approx=True,
    )


def test_cos():
    assert_equal(cos(0.5), pc.cos(0.5).as_py(), approx=True)
    assert_iterable_equal(
        cos(make_array([0.5, 0.6])), [pc.cos(0.5).as_py(), pc.cos(0.6).as_py()],
        approx=True,
    )


def test_cosh():
    assert_equal(cosh(0.5), 1.1276259652063807, approx=True)
    assert_iterable_equal(
        cosh(make_array([0.5, 0.6])),
        [1.1276259652063807, 1.1854652182422676],
        approx=True,
    )


def test_cospi():
    assert_equal(cospi(0.5), pc.cos(pi * 0.5).as_py(), approx=True)
    assert_iterable_equal(
        cospi(make_array([0.5, 0.6])),
        [pc.cos(pi * 0.5).as_py(), pc.cos(pi * 0.6).as_py()],
        approx=True,
    )


def test_sin():
    assert_equal(sin(0.5), pc.sin(0.5).as_py(), approx=True)
    assert_iterable_equal(
        sin(make_array([0.5, 0.6])), [pc.sin(0.5).as_py(), pc.sin(0.6).as_py()],
        approx=True,
    )


def test_sinh():
    assert_equal(sinh(0.5), 0.5210953054937474, approx=True)
    assert_iterable_equal(
        sinh(make_array([0.5, 0.6])),
        [0.5210953054937474, 0.6366535821482417],
        approx=True,
    )


def test_sinpi():
    assert_equal(sinpi(0.5), pc.sin(pi * 0.5).as_py(), approx=True)
    assert_iterable_equal(
        sinpi(make_array([0.5, 0.6])),
        [pc.sin(pi * 0.5).as_py(), pc.sin(pi * 0.6).as_py()],
        approx=True,
    )


def test_tan():
    assert_equal(tan(0.5), pc.tan(0.5).as_py(), approx=True)
    assert_iterable_equal(
        tan(make_array([0.5, 0.6])),
        [pc.tan(0.5).as_py(), pc.tan(0.6).as_py()],
        approx=True,
    )


def test_tanh():
    assert_equal(tanh(0.5), 0.46211715726000974, approx=True)
    assert_iterable_equal(
        tanh(make_array([0.5, 0.6])),
        [0.46211715726000974, 0.5370495669980353],
        approx=True,
    )


def test_tanpi():
    assert_equal(tanpi(0.5), pc.tan(pi * 0.5).as_py(), approx=True)
    assert_iterable_equal(
        tanpi(make_array([0.5, 0.6])),
        [pc.tan(pi * 0.5).as_py(), pc.tan(pi * 0.6).as_py()],
        approx=True,
    )


def test_atan2():
    assert_equal(atan2(0.5, 0.5), pc.atan2(0.5, 0.5).as_py(), approx=True)
    assert_iterable_equal(
        atan2(make_array([0.5, 0.6]), make_array([0.5, 0.6])),
        [pc.atan2(0.5, 0.5).as_py(), pc.atan2(0.6, 0.6).as_py()],
        approx=True,
    )
