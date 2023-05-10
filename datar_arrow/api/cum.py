import numpy as np
import pyarrow.compute as pc
from datar.apis.base import (
    cummax,
    cummin,
    cumprod,
    cumsum,
)
from ..utils import wrap_arrow_result, make_array


@cummax.register(object, backend="arrow")
def _cummax(x):
    return make_array(np.maximum.accumulate(x))


@cummin.register(object, backend="arrow")
def _cummin(x):
    return make_array(np.minimum.accumulate(x))


@cumprod.register(object, backend="arrow")
def _cumprod(x):
    return make_array(np.cumprod(x))


@cumsum.register(object, backend="arrow")
@wrap_arrow_result
def _cumsum(x):
    return pc.cumulative_sum(x)
