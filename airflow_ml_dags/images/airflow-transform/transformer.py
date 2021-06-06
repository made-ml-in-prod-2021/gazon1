"""Transform data before train of prediction"""

import os
import pickle
import logging
from enum import Enum, auto
from typing import NoReturn, Optional, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)


PARAMS = {
    'target': 'target',
    'num_features': [
        'age', 'sex', 'bmi', 'bp', 's1',
        's2', 's3', 's4', 's5', 's6'],
    'cat_features': [],
}


class MakeFeatureMode(Enum):
    """transform data for train/validation/testing"""
    train = auto()
    val = auto()
    test = auto()


@click.group()
def cli():
  pass


@click.command("fit_and_save_transformer")
@click.option("--input-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--transformer-dir", type=click.Path())
def fit_and_save_transformer(input_dir: str, output_dir: str,
                             transformer_dir: str):
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    X[PARAMS['target']] = y

    transformer = get_transformer(PARAMS)
    featured_df, target = make_features(
        transformer=transformer,
        df=X,
        params=PARAMS,
        mode=MakeFeatureMode.train,
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(transformer_dir, exist_ok=True)

    serialize_transformer(
        transformer,
        os.path.join(transformer_dir, "transformer.pkl"),
    )
    featured_df.to_csv(os.path.join(output_dir, "data.csv"),
                   header=True, index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"),
                   header=True, index=False)


@click.command("transform")
@click.option("--input-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--transformer-dir", type=click.Path(exists=True))
def apply_transformer(input_dir: str, output_dir: str, transformer_dir: str):
    transformer = deserialize_transformer(
        os.path.join(transformer_dir, "transformer.pkl"),
    )
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))


    if 'target.csv' in os.listdir(input_dir):
        mode=MakeFeatureMode.val
        X[PARAMS['target']] = pd.read_csv(
            os.path.join(input_dir, "target.csv"))
    else:
        mode=MakeFeatureMode.test

    featured_df, target = make_features(
        transformer=transformer,
        df=X,
        params=PARAMS,
        mode=mode,
    )

    os.makedirs(output_dir, exist_ok=True)
    featured_df.to_csv(os.path.join(output_dir, "data.csv"),
                   header=True, index=False)
    if mode is MakeFeatureMode.val:
        target.to_csv(os.path.join(output_dir, "target.csv"),
                      header=True, index=False)



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
    scaler = MinMaxScaler(feature_range=(-1, 1))
    pipeline = Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
    ])
    return pipeline

def get_transformer(params: dict) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("cat", get_cat_pipeline(), params['cat_features']),
            ("num", get_num_pipeline(), params['num_features']),
        ]
    )
    return transformer

def make_features(
    transformer: ColumnTransformer,
    df: pd.DataFrame,
    params: dict,
    mode: MakeFeatureMode = MakeFeatureMode.train,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    # ensure constant order of data columns
    # in training and testing
    if mode in (MakeFeatureMode.train, MakeFeatureMode.val) or \
       (mode is MakeFeatureMode.test and \
        params['target'] in df.columns):
        target = df.pop(params['target'])
    if mode is MakeFeatureMode.test and \
       params['target'] in df.columns:
        _ = df.pop(params['target'])
    if mode is MakeFeatureMode.train:
        transformer.fit(df)
    featured_df = pd.DataFrame(transformer.transform(df))
    if mode is MakeFeatureMode.test:
        return featured_df, None

    assert mode in (MakeFeatureMode.train, MakeFeatureMode.val)
    return featured_df, target

def serialize_transformer(transformer: ColumnTransformer,
                          path: str) -> NoReturn:
    with open(path, "wb") as f:
        pickle.dump(transformer, f)


def deserialize_transformer(path: str) -> ColumnTransformer:
    with open(path, "rb") as f:
        transformer = pickle.load(f)
    return transformer


cli.add_command(fit_and_save_transformer)
cli.add_command(apply_transformer)

if __name__ == "__main__":
    cli()
