from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..entities.split_params import SplitParams
from ..utils.utils import X_Pool


def data_split(
    processed_data: pd.DataFrame,
    params: SplitParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train, data_val = train_test_split(
        processed_data,
        test_size=params.val_size,
        random_state=params.random_state
    )
    return data_train, data_val
