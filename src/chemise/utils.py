from __future__ import annotations

import numpy as np


def make_metric_string(metrics: dict[str, str | np.ndarray | float], precision=4) -> str:
    """
    Make a string out of a metric dict
    :param metrics:
    :param precision:
    :return:
    """

    def value_format(v):
        if isinstance(v, str):
            return v
        if isinstance(v, np.ndarray):
            return [value_format(x) for x in v]
        try:
            fv = float(v)
            return f"{fv:.{precision}}"
        except TypeError:
            raise TypeError(f"Unsupported type ({type(v)}) to log")

    met_string = "{}: {}"
    return f"-- {', '.join([met_string.format(k, value_format(v)) for k, v in metrics.items()])}"


def seconds_pretty(seconds: float) -> str:
    """
    Format the number of seconds to a pretty string in most significant  e.g
    0.012 -> 12ms
    12.32 -> 12s
    :param seconds:
    :return: A string of the number of seconds in order of magnitude form
    """

    if seconds > 1:
        return f"{seconds:3.0f}s"

    second_exp = seconds
    for e in ["ms", "µs", "ns"]:
        second_exp = second_exp * 1e3
        if second_exp > 1:
            return f"{second_exp:3.0f}{e}"

    return f"{second_exp:3.3f}ns"


def mean_reduce_dicts(dict_list):
    transp = list_dict_to_dict_list(dict_list)
    res = {k: np.nanmean(np.stack(v), axis=0) for k, v in transp.items()}
    return res


def list_dict_to_dict_list(dict_list):
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    res = {k: [] for k in keys}
    for x in dict_list:
        [res[k].append(x[k]) for k in keys]
    return res


def datasetspec_to_zero(ds, batch_size: int = None, force_size: bool = False):
    """
    Convert a dataset elementSpec to `np.zeros` with the same shape
    :param ds: Data Spec
    :param batch_size: Default batch size to use if None
    :param force_size: Overwrite the batch size
    :return:
    """
    if isinstance(ds, tuple):
        return tuple(datasetspec_to_zero(el, batch_size=batch_size) for el in ds)

    if isinstance(ds, dict):
        return {k: datasetspec_to_zero(v, batch_size=batch_size) for k, v in ds.items()}

    # Replace batch size if None
    shape = ds.shape
    shape = shape[0] if (shape[0] and force_size) else batch_size, *shape[1:]
    return np.zeros(shape=shape, dtype=ds.dtype.as_numpy_dtype)
