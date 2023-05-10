import numpy as np
from datar.apis.base import (
    which,
    which_min,
    which_max,
)
from ..utils import make_array


@which.register(object, backend="arrow")
def _which(x):
    return make_array(np.flatnonzero(x))


@which_min.register(object, backend="arrow")
def _which_min(x):
    return make_array(np.argmin(x))


@which_max.register(object, backend="arrow")
def _which_max(x):
    return make_array(np.argmax(x))
