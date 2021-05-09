from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from src.entities.feature_params import FeatureParams
from src.entities.split_params import SplitParams
from src.entities.train_params import TrainParams
from src.entities.data_params import DataParams
from src.entities.general_params import GeneralParams


@dataclass()
class PipelineParams:
    feature: FeatureParams
    data_split: SplitParams
    data_load: DataParams
    model: TrainParams
    general: GeneralParams

PipelineParamsSchema = class_schema(PipelineParams)
