"""Date time functions"""
import datetime
import math

import pyarrow as pa
from datar.apis.base import as_date
from ..utils import make_array


@as_date.register(
    (
        pa.TimestampScalar,
        pa.Time64Scalar,
        pa.Time32Scalar,
        pa.Date64Scalar,
        pa.Date32Scalar,
    ),
    backend="arrow",
)
def _(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    return as_date(
        x.as_py(),
        format=format,
        try_formats=try_formats,
        optional=optional,
        tz=tz,
        origin=origin,
        __ast_fallback="normal",
        __backend="arrow",
    )


@as_date.register(datetime.date, backend="arrow")
def _as_date_d(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, int):
        tz = datetime.timedelta(hours=int(tz))

    return x + tz


@as_date.register(datetime.datetime, backend="arrow")
def _as_date_dt(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, int):
        tz = datetime.timedelta(hours=int(tz))

    return (x + tz).date()


@as_date.register(str, backend="arrow")
def _as_date_str(
    x: str,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, int):
        tz = datetime.timedelta(hours=int(tz))

    try_formats = try_formats or [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    if not format:
        format = try_formats
    else:
        format = [format]

    for fmt in format:
        try:
            return (datetime.datetime.strptime(x, fmt) + tz).date()
        except ValueError:
            continue

    if optional:
        return math.nan

    raise ValueError(
        "character string is not in a standard unambiguous format"
    )


@as_date.register(int, backend="arrow")
def _as_date_int(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, int):
        tz = datetime.timedelta(hours=int(tz))

    if isinstance(origin, str):
        origin = _as_date_str(origin)

    if origin is None:  # pragma: no cover
        origin = datetime.date(1969, 12, 31)

    dt = origin + datetime.timedelta(days=int(x)) + tz

    if isinstance(dt, datetime.datetime):
        return dt.date()
    return dt


@as_date.register((list, tuple, pa.Array), backend="arrow")
def _as_date_iter(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    return make_array(
        [
            as_date(
                el,
                format=format,
                try_formats=try_formats,
                optional=optional,
                origin=origin,
                tz=tz,
                __ast_fallback="normal",
                __backend="arrow",
            )
            for el in x
        ]
    )
