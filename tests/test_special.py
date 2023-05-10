import pytest  # noqa: F401

import pyarrow as pa
from datar.base import (
    beta,
    lgamma,
    digamma,
    trigamma,
    choose,
    factorial,
    gamma,
    lfactorial,
    lchoose,
    lbeta,
    psigamma,
)
from .utils import assert_equal, assert_iterable_equal


def test_beta():
    assert_equal(beta(1, 2), 0.5, approx=True)
    assert_iterable_equal(
        beta([1, 2], [2, 3]), [0.5, 0.08333333333333333], approx=True
    )
    assert_iterable_equal(
        beta(pa.array([1, 2]), pa.array([2, 3])),
        [0.5, 0.08333333333333333],
        approx=True,
    )


def test_lgamma():
    assert_equal(lgamma(1), 0, approx=True)
    assert_iterable_equal(lgamma([1, 2]), [0, 0], approx=True)
    assert_iterable_equal(lgamma(pa.array([1, 2])), [0, 0], approx=True)


def test_digamma():
    assert_equal(digamma(1), -0.5772156649015329, approx=True)
    assert_iterable_equal(
        digamma([1, 2]),
        [-0.5772156649015329, 0.42278433509846714],
        approx=True,
    )
    assert_iterable_equal(
        digamma(pa.array([1, 2])),
        [-0.5772156649015329, 0.42278433509846714],
        approx=True,
    )


def test_trigamma():
    assert_equal(trigamma(1), 1.6449340668482266, approx=True)
    assert_iterable_equal(
        trigamma([1, 2]), [1.6449340668482266, 0.6449340668482266], approx=True
    )
    assert_iterable_equal(
        trigamma(pa.array([1, 2])),
        [1.6449340668482266, 0.6449340668482266],
        approx=True,
    )


def test_choose():
    assert_equal(choose(2, 1), 2, approx=True)
    assert_iterable_equal(choose([2, 4], [1, 2]), [2, 6], approx=True)


def test_factorial():
    assert_equal(factorial(1), 1)
    assert_iterable_equal(factorial([1, 4]), [1, 24])


def test_gamma():
    assert_equal(gamma(1), 1)
    assert_iterable_equal(gamma([1, 2]), [1, 1])
    assert_iterable_equal(gamma(pa.array([1, 2])), [1, 1])


def test_lfactorial():
    assert_equal(lfactorial(1), 0)
    assert_iterable_equal(lfactorial([1, 2]), [0, 0.6931471805599453])
    assert_iterable_equal(
        lfactorial(pa.array([1, 2])), [0, 0.6931471805599453]
    )


def test_lchoose():
    assert_equal(lchoose(2, 1), 0.6931471805599453, approx=True)
    assert_iterable_equal(
        lchoose([2, 4], [1, 2]),
        [0.6931471805599453, 1.791759469228055],
        approx=True,
    )


def test_lbeta():
    assert_equal(lbeta(1, 2), -0.6931471805599453, approx=True)
    assert_iterable_equal(
        lbeta([1, 2], [2, 3]),
        [-0.6931471805599453, -2.4849066497880004],
        approx=True,
    )


def test_psigamma():
    assert_equal(psigamma(1, 1), 1.6449340668482266, approx=True)
    assert_iterable_equal(
        psigamma([1, 2], 1),
        [1.6449340668482266, 0.6449340668482266],
        approx=True,
    )
