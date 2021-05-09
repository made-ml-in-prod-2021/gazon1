import logging
import sys
import pickle
import pandas as pd
import numpy as np
from catboost import (
    CatBoost, Pool, CatBoostClassifier,
)

def X_Pool(data: pd.DataFrame, label: np.array):
    return Pool(
        data=data,
        label=label,
    )

def serialize_model(model, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

def deserialize_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def setup_logger(
    level = logging.DEBUG,
    **kwargs
):
    """Log to console, file"""
    formatter = logging.Formatter(
        '\r[%(asctime)s] %(levelname)s: %(message)s', datefmt="%H:%M:%S")

    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    # file_handler = logging.FileHandler(filename=log_file)
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
