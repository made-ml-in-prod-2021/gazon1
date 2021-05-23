import gc
import logging
import os
import pickle
import sys
from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, FileType, Namespace)
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

sys.path.append("/home/max/MADE/ml-prod/gazon1/")

import json
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.data import load_data
from src.data_split.data_split import data_split
from src.entities.pipeline_params import PipelineParams, PipelineParamsSchema
from src.evaluate.evaluate import evaluate
from src.features.features import (MakeFeatureMode, deserialize_transformer,
                                   make_features)
from src.train.train import train_model
from src.utils.utils import X_Pool, deserialize_model, setup_logger

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
