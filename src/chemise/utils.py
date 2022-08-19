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
        try:
            fv = float(v)
            return f"{fv:.{precision}}"
        except TypeError:
            raise TypeError("Can only log scaler variables")

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
    res = {k: np.mean(v) for k, v in transp.items()}
    return res


def list_dict_to_dict_list(dict_list):
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    res = {k: [x[k] for x in dict_list] for k in keys}
    return res

def datasetspec_to_zero(ds):
    """
    Convert a dataset elementSpec to `np.zeros` with the same shape
    :param ds:
    :return:
    """
    if isinstance(ds, tuple):
        return tuple(datasetspec_to_zero(el) for el in ds)

    if isinstance(ds, dict):
        return {k: datasetspec_to_zero(v) for k, v in ds.items()}

    return np.zeros(shape=ds.shape, dtype=ds.dtype.as_numpy_dtype)