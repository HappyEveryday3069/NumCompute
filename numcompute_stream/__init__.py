"""
NumCompute-Stream: a modularised, decision tree-based streaming ML framework.

Only NumPy and matplotlib are used. See ``README.md`` for a tour of the modules.
"""

__version__ = "0.2.0"

from .io import read_csv, read_csv_chunks
from .preprocessing import (Imputer, MinMaxScaler, OneHotEncoder, SimpleImputer,
                            StandardScaler)
from .stats import RunningStats, StreamingHistogram, StreamingQuantiles
from .metrics import (StreamingAccuracy, StreamingAUC, StreamingConfusionMatrix,
                      StreamingF1, StreamingMSE, StreamingPrecision, StreamingRecall)
from .tree import DecisionTreeClassifier
from .pipeline import Pipeline
from .stream import StreamTrainer
from .utils import NotFittedError, iter_chunks
