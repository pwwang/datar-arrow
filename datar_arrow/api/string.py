from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datar.core.utils import logger
from datar.apis.base import (
    grep,
    grepl,
    sub,
    gsub,
    strsplit,
    paste,
    paste0,
    sprintf,
    substr,
    substring,
    startswith,
    endswith,
    strtoi,
    trimws,
    toupper,
    tolower,
    chartr,
    nchar,
    nzchar,
)
from ..utils import (
    broadcast_arrays,
    is_null,
    is_scalar,
    make_array,
    wrap_arrow_result,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..arrow_ext import DatarArray


def _warn_more_pat_or_rep(pattern, fun, arg="pattern"):
    """Warn when there are more than one pattern or replacement provided"""
    if is_scalar(pattern):
        return pattern
    if len(pattern) == 1:
        return pattern[0]

    logger.warning(
        "[datar_arrow] "
        "In %s(...), argument `%s` has length > 1 and only the "
        "first element will be used",
        fun,
        arg,
    )
    return pattern[0]


def _match(
    text: DatarArray,
    pattern: str,
    ignore_case: bool,
    invert: bool,
    fixed: bool,
) -> pa.BooleanArray:
    """Do the regex match"""
    if fixed:
        out = pc.match_substring(
            text.storage,
            pattern,
            ignore_case=ignore_case,
        )
    else:
        out = pc.match_substring_regex(
            text.storage,
            pattern,
            ignore_case=ignore_case,
        )

    return pc.invert(out) if invert else out


def _sub_(
    pattern: str,
    replacement: str,
    x: DatarArray,
    ignore_case: bool = False,
    fixed: bool = False,
    count: int = 1,
    fun: str = "sub",
) -> pa.StringArray:
    """Replace a pattern with replacement for elements in x,
    with argument count available
    """
    if ignore_case:
        raise NotImplementedError("ignore_case is not supported yet")

    pattern = _warn_more_pat_or_rep(pattern, fun)
    replacement = _warn_more_pat_or_rep(replacement, fun, "replacement")
    x = make_array(x, dtype="str")
    if fixed:
        return pc.replace_substring(
            x.storage,
            pattern,
            replacement,
            max_replacements=count,
        )

    return pc.replace_substring_regex(
        x.storage,
        pattern,
        replacement,
        max_replacements=count,
    )


def _prepare_nchar(x, type_, keep_na):
    """Prepare arguments for n(z)char"""
    if type_ not in ["chars", "bytes", "width"]:
        raise ValueError(
            f"Invalid type argument, expect 'chars', 'bytes' or 'width', "
            f"got {type_}"
        )
    if keep_na is None:
        keep_na = type != "width"

    return x, keep_na


@np.vectorize
def _nchar_(x, retn, allow_na, keep_na, na_len):
    """Get the size of a scalar string"""
    if is_null(x):
        return None if keep_na else na_len

    if retn == "width":
        try:
            from wcwidth import wcswidth
        except ImportError as imperr:  # pragma: no cover
            raise ImportError(
                "`nchar(x, type='width')` requires `wcwidth` package.\n"
                "Try: pip install -U wcwidth"
            ) from imperr

        return wcswidth(x)
    if retn == "chars":
        return len(x)

    if isinstance(x, bytes):
        return len(x)

    try:
        x = x.encode("utf-8")
    except UnicodeEncodeError:  # pragma: no cover
        if allow_na:
            return None
        raise
    return len(x)


@grep.register(object, backend="arrow")
def _grep(
    pattern,
    x,
    ignore_case=False,
    value=False,
    fixed=False,
    invert=False,
):
    pattern = _warn_more_pat_or_rep(pattern, "grepl")
    x_scalar = is_scalar(x)
    x = make_array(x, dtype=str)
    matched = _match(
        x,
        pattern,
        ignore_case=ignore_case,
        invert=invert,
        fixed=fixed,
    )
    matched = np.flatnonzero(matched)
    out = x.take(matched) if value else make_array(matched)
    return out[0] if x_scalar and len(out) > 0 else out


@grepl.register(object, backend="arrow")
@wrap_arrow_result
def _grepl(
    pattern,
    x,
    ignore_case=False,
    fixed=False,
    invert=False,
):
    pattern = _warn_more_pat_or_rep(pattern, "grepl")
    out = _match(
        make_array(x, dtype=str),
        pattern,
        ignore_case=ignore_case,
        invert=invert,
        fixed=fixed,
    )
    return out[0] if is_scalar(x) else out


@sub.register(object, backend="arrow")
@wrap_arrow_result
def _sub(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
):
    out = _sub_(pattern, replacement, x, ignore_case, fixed, 1, "sub")
    return out[0] if is_scalar(x) else out


@gsub.register(object, backend="arrow")
@wrap_arrow_result
def _gsub(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
):
    out = _sub_(pattern, replacement, x, ignore_case, fixed, None, "gsub")
    return out[0] if is_scalar(x) else out


@strsplit.register(object, backend="arrow")
@wrap_arrow_result
def _strsplit(x, split, fixed=False) -> pa.ListArray:
    x = make_array(x, dtype=str)
    if fixed:
        out = pc.split_pattern(x.storage, split)
    else:
        out = pc.split_pattern_regex(x.storage, split)

    return out


@paste.register(object, backend="arrow")
@wrap_arrow_result
def _paste(*args, sep=" ", collapse=None):
    arrs = broadcast_arrays(*args)
    out = pc.binary_join_element_wise(
        *(a.storage.cast("str") for a in arrs),
        sep,
        null_handling="skip",
    )
    if collapse is None:
        return out

    return collapse.join(make_array(out))


@paste0.register(object, backend="arrow")
def _paste0(*args, collapse=None):
    return _paste(*args, sep="", collapse=collapse)


@sprintf.register(object, backend="arrow")
def _sprintf(fmt, *args):
    return make_array(
        np.vectorize(lambda fmt, *args: fmt % args)(
            *np.broadcast_arrays(fmt, *args)
        ),
        dtype=str,
    )


@substr.register(object, backend="arrow")
@wrap_arrow_result
def _substr(x, start, stop):
    return pc.utf8_slice_codeunits(x, start, stop)


@substring.register(object, backend="arrow")
@wrap_arrow_result
def _substring(x, first, last=1000000):
    return pc.utf8_slice_codeunits(x, first, last)


@startswith.register(object, backend="arrow")
@wrap_arrow_result
def _startswith(x, prefix):
    return pc.starts_with(x, prefix)


@endswith.register(object, backend="arrow")
@wrap_arrow_result
def _endswith(x, suffix):
    return pc.ends_with(x, suffix)


@strtoi.register(object, backend="arrow")
@wrap_arrow_result
def _strtoi(x, base=0):
    if base not in (0, 10):
        raise ValueError("`base` other than 0 or 10 not supported")
    x_scalar = is_scalar(x)
    x = make_array(x, dtype=str)
    x = pa.array(
        list(x),
        type="string",
        mask=pc.invert(pc.utf8_is_digit(x.storage)),
    )
    out = x.cast("int64")
    return out[0] if x_scalar else out


@trimws.register(object, backend="arrow")
@wrap_arrow_result
def _trimws(x, which="both", whitespace=r" \t"):
    if which == "both":
        return pc.utf8_trim(x, whitespace)
    if which == "left":
        return pc.utf8_ltrim(x, whitespace)
    if which == "right":
        return pc.utf8_rtrim(x, whitespace)
    raise ValueError("`which` must be one of 'both', 'left', 'right'")


@toupper.register(object, backend="arrow")
@wrap_arrow_result
def _toupper(x):
    return pc.utf8_upper(x)


@tolower.register(object, backend="arrow")
@wrap_arrow_result
def _tolower(x):
    return pc.utf8_lower(x)


@chartr.register(object, backend="arrow")
@wrap_arrow_result
def _chartr(old, new, x):
    x_scalar = is_scalar(x)
    x = make_array(x, dtype=str)
    old = _warn_more_pat_or_rep(old, "chartr", "old")
    new = _warn_more_pat_or_rep(new, "chartr", "new")

    new = new[: len(old)]
    for oldc, newc in zip(old, new):
        x = pc.replace_substring(x.storage, oldc, newc)
    return x[0] if x_scalar else x


@nchar.register(object, backend="arrow")
def _nchar(
    x,
    type_: str = "width",
    allow_na: bool = True,
    keep_na: bool = False,
    _na_len: int = 2,
):
    x_scalar = is_scalar(x)
    x, keep_na = _prepare_nchar(x, type_, keep_na)
    out = make_array(
        _nchar_(
            make_array(x),
            retn=type_,
            allow_na=allow_na,
            keep_na=keep_na,
            na_len=_na_len,
        )
    )
    return out[0] if x_scalar else out


@nzchar.register(object, backend="arrow")
@wrap_arrow_result
def _nzchar(x, keep_na: bool = False):
    x = make_array(x, dtype=str)
    x = pc.invert(pc.match_like(x.storage, ""))
    return x if keep_na else x.fill_null(True)
