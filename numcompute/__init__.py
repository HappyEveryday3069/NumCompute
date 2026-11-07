"""
NumCompute: A High-Performance Numpy Toolkit.
"""


__version__ = "0.0.1"

from .io import read_csv, read_csv_chunks
from .preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
from .sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search