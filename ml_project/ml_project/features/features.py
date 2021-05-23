import logging
import pickle
from enum import Enum, auto
from typing import NoReturn, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..entities.feature_params import FeatureParams

logger = logging.getLogger(__name__)


class MakeFeatureMode(Enum):
    train = auto()
    val = auto()
    test = auto()


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

def make_features(
    transformer: ColumnTransformer,
    df: pd.DataFrame,
    params: FeatureParams,
    mode: MakeFeatureMode = MakeFeatureMode.train,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    # import pdb; pdb.set_trace()

    # ensure constant order of data columns
    # in training and testing
    # df = df.reindex(sorted(df.columns), axis=1)

    if mode in (MakeFeatureMode.train, MakeFeatureMode.val) or \
       (mode is MakeFeatureMode.test and \
        params.target in df.columns):
        target = df.pop(params.target)
    if mode is MakeFeatureMode.test and \
       params.target in df.columns:
        _ = df.pop(params.target)
    if mode is MakeFeatureMode.train:
        transformer.fit(df)
    featured_df = pd.DataFrame(transformer.transform(df))
    if mode is MakeFeatureMode.test:
        return featured_df, None

    assert mode in (MakeFeatureMode.train, MakeFeatureMode.val)
    return featured_df, target

def serialize_transformer(
        transformer: ColumnTransformer, path: str) -> NoReturn:
    with open(path, "wb") as f:
        pickle.dump(transformer, f)


def deserialize_transformer(path: str) -> ColumnTransformer:
    with open(path, "rb") as f:
        transformer = pickle.load(f)
    return transformer
