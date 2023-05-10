import pyarrow as pa
from datar.core.plugin import plugin

# For simplug to retrieve the version
from .version import __version__  # noqa: F401
from .utils import flatten_slice, make_array

priority = -1


@plugin.impl
def base_api():
    from .api import (  # noqa: F401
        arithm,
        asis,
        bessel,
        # complex,
        constants,
        cum,
        date,
        factor,
        random,
        seq,
        sets,
        special,
        string,
        trig,
        which,
    )

    return {
        "pi": constants.pi,
        "letters": constants.letters,
        "LETTERS": constants.LETTERS,
        "month_abb": constants.month_abb,
        "month_name": constants.month_name,
        "NaN": constants.NaN,
        "Inf": constants.Inf,
        "NA": constants.NA,
        "NULL": constants.NULL,
    }


@plugin.impl
def get_versions():
    return {
        "datar-arrow": __version__,
        "pyarrow": pa.__version__,
    }


@plugin.impl
def c_getitem(item):
    if isinstance(item, slice):
        return flatten_slice(item)

    elif isinstance(item, tuple):
        return make_array(
            pa.concat_arrays(
                [
                    flatten_slice(i).storage
                    if isinstance(i, slice)
                    else make_array(i).storage
                    for i in item
                ]
            )
        )

    return item
