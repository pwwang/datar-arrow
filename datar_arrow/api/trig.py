import numpy as np
import pyarrow.compute as pc
from datar.apis.base import (
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
)
from ..utils import wrap_arrow_result
from ..arrow_ext import DatarArray


@acos.register(object, backend="arrow")
@wrap_arrow_result
def _acos(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.acos(x)


@acosh.register(object, backend="arrow")
@wrap_arrow_result
def _acosh(x):
    if isinstance(x, DatarArray):
        x = x.storage
    # ln(x + sqrt(x^2 - 1))
    return pc.ln(
        pc.add(
            x,
            pc.multiply(
                pc.sqrt(pc.subtract(x, 1)),
                pc.sqrt(pc.add(x, 1)),
            ),
        )
    )


@asin.register(object, backend="arrow")
@wrap_arrow_result
def _asin(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.asin(x)


@asinh.register(object, backend="arrow")
@wrap_arrow_result
def _asinh(x):
    if isinstance(x, DatarArray):
        x = x.storage
    # ln(x + sqrt(x^2 + 1))
    return pc.ln(
        pc.add(
            x,
            pc.sqrt(pc.add(pc.multiply(x, x), 1)),
        )
    )


@atan.register(object, backend="arrow")
@wrap_arrow_result
def _atan(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.atan(x)


@atanh.register(object, backend="arrow")
@wrap_arrow_result
def _atanh(x):
    if isinstance(x, DatarArray):
        x = x.storage
    # 0.5 * ln((1 + x) / (1 - x))
    return pc.multiply(
        0.5,
        pc.ln(pc.divide(pc.add(1, x).cast("double"), pc.subtract(1, x))),
    )


@cos.register(object, backend="arrow")
@wrap_arrow_result
def _cos(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.cos(x)


@cosh.register(object, backend="arrow")
@wrap_arrow_result
def _cosh(x):
    if isinstance(x, DatarArray):
        x = x.storage
    # (e^x + e^-x) / 2
    return pc.divide(
        pc.add(pc.exp(x), pc.exp(pc.multiply(-1, x))),
        2.0,
    )


@cospi.register(object, backend="arrow")
@wrap_arrow_result
def _cospi(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.cos(pc.multiply(np.pi, x))


@sin.register(object, backend="arrow")
@wrap_arrow_result
def _sin(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.sin(x)


@sinh.register(object, backend="arrow")
@wrap_arrow_result
def _sinh(x):
    if isinstance(x, DatarArray):
        x = x.storage
    # (e^x - e^-x) / 2
    return pc.divide(
        pc.subtract(pc.exp(x), pc.exp(pc.multiply(-1, x))),
        2.0,
    )


@sinpi.register(object, backend="arrow")
@wrap_arrow_result
def _sinpi(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.sin(pc.multiply(np.pi, x))


@tan.register(object, backend="arrow")
@wrap_arrow_result
def _tan(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.tan(x)


@tanh.register(object, backend="arrow")
@wrap_arrow_result
def _tanh(x):
    if isinstance(x, DatarArray):
        x = x.storage
    # (e^x - e^-x) / (e^x + e^-x)
    return pc.divide(
        pc.subtract(pc.exp(x), pc.exp(pc.multiply(-1, x))).cast("double"),
        pc.add(pc.exp(x), pc.exp(pc.multiply(-1, x))),
    )


@tanpi.register(object, backend="arrow")
@wrap_arrow_result
def _tanpi(x):
    if isinstance(x, DatarArray):
        x = x.storage
    return pc.tan(pc.multiply(np.pi, x))


@atan2.register(object, backend="arrow")
@wrap_arrow_result
def _atan2(y, x):
    if isinstance(x, DatarArray):
        x = x.storage
    if isinstance(y, DatarArray):
        y = y.storage
    return pc.atan2(y, x)
