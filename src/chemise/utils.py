import numpy as np

def reduce_dicts(dict_list):
    keys = dict_list[0].keys()
    res = {k: np.mean([x[k] for x in dict_list]) for k in keys}
    return res

def list_dict_to_dict_list(dict_list):
    keys = dict_list[0].keys()
    res = {k: [x[k] for x in dict_list] for k in keys}
    return res