import gc
import logging
import os
import pickle
from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, FileType, Namespace)
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoost, CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

import pickle
from typing import Dict, Union

import catboost
import numpy as np
import pandas as pd
from src.entities.feature_params import FeatureParams
from src.entities.train_params import TrainParams


def train_model(
    X_train: catboost.Pool, X_val: catboost.Pool,
    train_params: TrainParams,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**train_params.__dict__)
    model.fit(X_train, eval_set=X_val)
    return model

