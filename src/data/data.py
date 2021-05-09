import gc
import logging
import os
import pickle
import sys
from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, FileType, Namespace)
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Text

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

def load_data(path: Text) -> pd.DataFrame:
    return pd.read_csv(path)
