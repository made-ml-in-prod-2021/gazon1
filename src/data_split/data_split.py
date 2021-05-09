from typing import Tuple
import pandas as pd
from argparse import (
    ArgumentDefaultsHelpFormatter, ArgumentParser,
    ArgumentTypeError, FileType, Namespace
)

import os

import yaml
from sklearn.model_selection import train_test_split
from src.utils.utils import (
    X_Pool,
)
from src.entities.split_params import SplitParams

def data_split(
    processed_data: pd.DataFrame,
    params: SplitParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # X_train, X_test, y_train, y_test = train_test_split(
        # processed_data,
        # test_size=params.val_size,
        # random_state=params.random_state
    # )
    # train_pool = X_Pool(X_train, y_train)
    # val_pool = X_Pool(X_val, y_val)
    # return train_pool, val_pool

    data_train, data_val = train_test_split(
        processed_data,
        test_size=params.val_size,
        random_state=params.random_state
    )
    return data_train, data_val


# def callback(args):
#     config = yaml.safe_load(open(args.config_path))

#     data = data_split(config)
#     save_data(data, config)

# def setup_parser(parser : ArgumentParser):
#     parser.add_argument("--config", dest="config_path", required=True)
#     parser.set_defaults(callback=callback)

# if __name__ == "__main__":
#     parser = ArgumentParser(
#         prog = 'split-data-for-catboost-into-folds',
#         description='Split catboost dataset into folds',
#         formatter_class=ArgumentDefaultsHelpFormatter
#     )

#     setup_parser(parser)
#     arguments = parser.parse_args()
#     arguments.callback(arguments)
