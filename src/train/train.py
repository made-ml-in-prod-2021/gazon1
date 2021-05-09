from argparse import (
    ArgumentDefaultsHelpFormatter, ArgumentParser,
    ArgumentTypeError, FileType, Namespace
)
from copy import deepcopy
import os
import pickle
import gc
import logging

import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import (
    CatBoost, Pool, CatBoostClassifier,
)

logger = logging.getLogger(__name__)

import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
import catboost
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.entities.feature_params import FeatureParams
from src.entities.train_params import TrainParams

# ValidModelClass = Union[CatBoostClassifier, CatBoostRegressor]

def train_model(
    X_train: catboost.Pool, X_val: catboost.Pool,
    train_params: TrainParams,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**train_params.__dict__)
    model.fit(X_train, eval_set=X_val)
    return model


# def train_model(
#     config: dict,
#     **kwargs,
# ):
#     if 'X_train' in kwargs:
#         X_train = kwargs['X_train']
#     else:
#         catboost_train_dataset_path = \
#             'quantized://' + \
#             os.path.join(
#                 config['base']['dir'],
#                 config['data_load']['dataset_train']
#             )
#         X_train = Pool(catboost_train_dataset_path)

#     if 'X_val' in kwargs:
#         X_val = kwargs['X_val']
#     else:
#         catboost_val_dataset_path = \
#             'quantized://' + \
#             os.path.join(
#                 config['base']['dir'],
#                 config['data_load']['dataset_val']
#             )
#         X_val = Pool(catboost_val_dataset_path)


#     params = {
#         'loss_function': config['model']['loss_function'],
#         'custom_metric': [
#             'AUC:type=Ranking',
#             'MAP:top=100',
#             'PrecisionAt:top=100',
#             'MAP:top=20',
#             'PrecisionAt:top=20'
#         ],
#         'verbose': config['model']['verbose'],
#         'random_strength': config['model']['random_strength'],
#         'bootstrap_type': config['model']['bootstrap_type'],
#         'bagging_temperature': config['model']['bagging_temperature'],
#         'n_estimators': config['model']['n_estimators'],
#     }
#     model = CatBoost(params=params)
#     model.fit(
#         X_train,
#         eval_set = X_val,
#     )

#     logger.info('Оценка модели на тестовой выборке')
#     d = model.eval_metrics(
#         X_train, metrics=['AUC:type=Ranking', 'MAP:top=100'],
#         eval_period=model.tree_count_
#     )
#     logger.info(f"AUC = {d['AUC:type=Ranking']}")
#     logger.info(f"MAP = {d['MAP:top=100']}")
#     return model

# def callback(args):
#     config = yaml.safe_load(open(args.config_path))
#     model = train_model(config)

#     # save model
#     model_path = \
#         os.path.join(
#             config['base']['dir'],
#             config['train']['model_path'],
#         )

#     model.save_model(
#         model_path,
#         format="cbm",
#         export_parameters=None,
#         pool=None,
#     )

# def setup_parser(parser : ArgumentParser):
#     parser.add_argument("--config", dest="config_path", required=True)
#     parser.set_defaults(callback=callback)

# if __name__ == "__main__":
#     parser = ArgumentParser(
#         prog = 'train-catboost',
#         description='Train catboost on preprocessed dataset',
#         formatter_class=ArgumentDefaultsHelpFormatter
#     )

# setup_parser(parser)
# arguments = parser.parse_args()
# arguments.callback(arguments)
