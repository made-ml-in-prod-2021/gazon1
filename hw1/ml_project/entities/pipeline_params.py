from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplitParams
from .train_params import TrainParams
from .data_params import DataParams
from .general_params import GeneralParams


@dataclass()
class PipelineParams:
    feature: FeatureParams
    data_split: SplitParams
    data_load: DataParams
    model: TrainParams
    general: GeneralParams

PipelineParamsSchema = class_schema(PipelineParams)
