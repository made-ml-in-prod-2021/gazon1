from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    """
    list of features to process differently:
    - categorica
    - numerical
    - to drop features or not
    - target column
    """
    cat_features: List[str]
    num_features: List[str]
    to_drop: Optional[List[str]]
    target: Optional[str]
    transformer_save_path: str
