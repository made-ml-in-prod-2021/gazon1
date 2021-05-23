import logging
from typing import Text

import pandas as pd

logger = logging.getLogger(__name__)

def load_data(path: Text) -> pd.DataFrame:
    return pd.read_csv(path)
