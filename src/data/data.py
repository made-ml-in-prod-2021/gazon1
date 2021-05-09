import gc
import pickle
import logging
from typing import Text

from copy import deepcopy
from datetime import datetime, timedelta
from argparse import (
    ArgumentDefaultsHelpFormatter, ArgumentParser,
    ArgumentTypeError, FileType, Namespace
)

import yaml
import numpy as np
import pandas as pd

import sys, os

logger = logging.getLogger(__name__)

def load_data(path: Text) -> pd.DataFrame:
    return pd.read_csv(path)

# def main(args: Namespace):
#     config = yaml.safe_load(open(args.config_path))
#     load_path = os.path.join(
#         config['base']['dir'],
#         config['data']['data_raw']['dir'],
#         config['data']['data_raw']['file_name'],
#     )
#     return load_data(load_path)

# def setup_parser(parser : ArgumentParser):
#     parser.add_argument("--config", dest="config_path", required=True)
#     parser.set_defaults(callback=main)

# if __name__ == "__main__":
#     parser = ArgumentParser(
#         prog = 'download-data-for-catboost',
#         description='download data for catboost',
#         formatter_class=ArgumentDefaultsHelpFormatter
#     )

#     setup_parser(parser)
#     arguments = parser.parse_args()
#     arguments.callback(arguments)
