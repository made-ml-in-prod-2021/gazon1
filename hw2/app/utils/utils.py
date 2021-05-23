import os
from typing import Dict, Union, Text, List
import sys
import logging
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from fastapi import HTTPException
from pydantic import BaseModel, validator

logger = logging.getLogger()

def setup_logger(level = logging.DEBUG, **kwargs):
    """Log to console, file"""
    formatter = logging.Formatter(
        '\r[%(asctime)s] %(levelname)s: %(message)s', datefmt="%H:%M:%S")

    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class Model:
    def __init__(self, transformer_path: Text,
                 model_path: Text):
        self.transformer_path = transformer_path
        self.model_path = model_path
        self.model = None
        self.transformer = None

    def _deserialize(self, path: str):
        with open(path, "rb") as f:
            unpickled = pickle.load(f)
        return unpickled

    def _deserialize_transformer(self, path: str) -> ColumnTransformer:
        return self._deserialize(path)

    def _deserialize_model(self, path: str) -> CatBoostClassifier:
        return self._deserialize(path)

    def load(self,):
        try:
            logger.info(f'load transformer from {self.transformer_path}')
            self.transformer = self._deserialize_transformer(
                self.transformer_path)
            logger.info(f'load model from {self.model_path}')
            self.model = self._deserialize_model(
                self.model_path)
        except FileNotFoundError as e:
            logger.error(e)
            return

    def predict(
        self,
        data: pd.DataFrame,
    ) -> float:
        col_names = set([
            'cp', 'restecg', 'exang', 'thal',
            'ca', 'slope', 'sex', 'fbs', 'age',
            'trestbps', 'chol', 'thalach', 'oldpeak'
        ])
        given_columns = set(data.columns)
        assert col_names == given_columns, \
            f"{col_names - given_columns}, {given_columns - col_names}"
        # logger.info(f"feature names pf tr: {self.transformer.get_feature_names()}")
        transformed_data = self.transformer.transform(data)
        return self.model.predict_proba(transformed_data)[0, 1]


def generate_example_from_real_dataset(
    real_dataset_filepath: str,
) -> Dict[str, Union[int, float]]:
    """Random select of one item from read dataset
        and return features in HeartRequestModel format
    """
    if not os.path.exists(real_dataset_filepath):
        raise ValueError("The dataset filepath does not exist")
        return
    df = pd.read_csv(real_dataset_filepath)
    df.pop("target")
    single_example = list(
        df.sample(n=1, replace=True).to_dict(orient='index').values()
    )[0]
    return single_example
