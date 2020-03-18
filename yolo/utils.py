import numpy as np


def decode_yaml_tuple(tuple_str):
    return np.array(list(map(lambda x: list(map(int, str.split(x, ','))), tuple_str.split())))
