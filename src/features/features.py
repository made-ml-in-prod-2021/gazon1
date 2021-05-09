import gc
import logging
import os
import pickle
from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, FileType, Namespace)
from copy import deepcopy
from datetime import datetime
from enum import Enum, auto
from typing import NoReturn, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.entities.feature_params import FeatureParams

logger = logging.getLogger(__name__)


def get_cat_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    encoder = OneHotEncoder()
    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("encoder", encoder),
        ]
    )
    return pipeline


def get_num_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    pipeline = Pipeline(
        [
            ("imputer", imputer),
        ]
    )
    return pipeline

def get_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("cat", get_cat_pipeline(), params.cat_features),
            ("num", get_num_pipeline(), params.num_features),
        ]
    )
    return transformer



class MakeFeatureMode:
    train = auto()
    val = auto()
    test = auto()

def make_features(
    transformer: ColumnTransformer,
    df: pd.DataFrame,
    params: FeatureParams,
    mode: MakeFeatureMode = MakeFeatureMode.train,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if mode is MakeFeatureMode.train:
        transformer.fit(df)
    featured_df = pd.DataFrame(transformer.transform(df))
    if mode is MakeFeatureMode.test:
        return featured_df, None

    target = df[params.target]
    return featured_df, target

def serialize_transformer(
        transformer: ColumnTransformer, path: str) -> NoReturn:
    with open(path, "wb") as f:
        pickle.dump(transformer, f)


def deserialize_transformer(path: str) -> ColumnTransformer:
    with open(path, "rb") as f:
        transformer = pickle.load(f)
    return transformer
