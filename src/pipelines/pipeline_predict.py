import gc
import pickle
import logging
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

# sys.path = [s for s in sys.path if '/als' not in s]
sys.path.append("/home/max/MADE/ml-prod/gazon1/")

import json
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.data import load_data
from src.entities.pipeline_params import (
    PipelineParams, PipelineParamsSchema
)
from src.features.features import (
    make_features,
    deserialize_transformer,
    MakeFeatureMode,
)

from src.data_split.data_split import data_split
from src.train.train import train_model
from src.utils.utils import (
    setup_logger,
    deserialize_model,
    X_Pool,
)
from src.evaluate.evaluate import evaluate

setup_logger()
logger = logging.getLogger(__name__)


def predict_pipeline(pipeline_params: PipelineParams) -> None:
    logger.info('load data')

    raw_dir = os.path.join(
        pipeline_params.general.dir,
        pipeline_params.data_load.dataset_raw_dir,
    )
    data_test = pd.read_pickle(os.path.join(
        raw_dir,
        pipeline_params.data_load.dataset_raw_test,
    ))

    logger.info('preprocess test dataset')
    data_transformer = deserialize_transformer(
        os.path.join(
            pipeline_params.general.dir,
            pipeline_params.feature.transformer_save_path,
    ))
    X_test, _ = make_features(
        data_transformer, data_test,
        pipeline_params.feature,
        mode=MakeFeatureMode.test,
    )

    proccesed_dir = os.path.join(
        pipeline_params.general.dir,
        pipeline_params.data_split.dataset_processed_dir,
    )
    model = deserialize_model(os.path.join(
        pipeline_params.general.dir,
        pipeline_params.general.model_path,
    ))
    X_test['predictions'] = model.predict(X_test)
    X_test.to_csv(os.path.join(
        proccesed_dir,
        pipeline_params.data_split.dataset_processed_test,
    ), index=False)

@hydra.main(config_path="../../config", config_name="params.yaml")
def main(config: DictConfig) -> None:
    """
    Hydra wrapper for parsing CLI arguments
    :return: Nothing
    """
    # os.chdir(hydra.utils.to_absolute_path("."))
    schema = PipelineParamsSchema()
    logger.info(f"Pipeline config:\n{OmegaConf.to_yaml(config)}")
    config = schema.load(config)
    predict_pipeline(config)


if __name__ == "__main__":
    main()
