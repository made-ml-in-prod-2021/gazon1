import logging
import os

import hydra
from first_hw.src.data.data import load_data
from first_hw.src.data_split.data_split import data_split
from first_hw.src.entities.pipeline_params import (PipelineParams,
                                                   PipelineParamsSchema)
from first_hw.src.evaluate.evaluate import evaluate
from first_hw.src.features.features import (MakeFeatureMode, get_transformer,
                                            make_features,
                                            serialize_transformer)
from first_hw.src.train.train import train_model
from first_hw.src.utils.utils import X_Pool, serialize_model, setup_logger
from omegaconf import DictConfig, OmegaConf

setup_logger()
logger = logging.getLogger(__name__)

def train_pipeline(pipeline_params: PipelineParams) -> None:
    """
    1. Загружаем данные
    2. Разделяем на train/val
    3. Препроцессинг
    4. Обучение модели
    5. оценка на валидации
    4. сохранение модели и предсказание/ оценка на валидации
    """
    logger.info('load data')
    raw_dir = os.path.join(
        pipeline_params.general.dir,
        pipeline_params.data_load.dataset_raw_dir,
    )

    dataset_raw_path = os.path.join(
        raw_dir,
        pipeline_params.data_load.dataset_raw_file,
    )
    data = load_data(dataset_raw_path)

    logger.info('split data into train and validation datasets')
    data_train, data_val = data_split(data, pipeline_params.data_split)

    logger.info('saving raw training and validation datasets for preidctions')

    data_train.to_pickle(os.path.join(
        raw_dir,
        pipeline_params.data_load.dataset_raw_train,
    ))

    data_val.to_pickle(os.path.join(
        raw_dir,
        pipeline_params.data_load.dataset_raw_val,
    ))

    logger.info('preprocess train dataset')
    data_transformer = get_transformer(pipeline_params.feature)
    X_train, y_train = make_features(
        data_transformer, data_train,
        pipeline_params.feature,
        mode=MakeFeatureMode.train,
    )
    serialize_transformer(
        data_transformer,
        os.path.join(
            pipeline_params.general.dir,
            pipeline_params.feature.transformer_save_path,
    ))

    logger.info('preprocess validation dataset')
    X_val, y_val = make_features(
        data_transformer,
        data_val,
        pipeline_params.feature,
        mode=MakeFeatureMode.val,
    )

    logger.info('train model')

    X_train_pool = X_Pool(X_train, y_train)
    X_val_pool = X_Pool(X_val, y_val)
    model = train_model(
        X_train_pool,
        X_val_pool,
        pipeline_params.model
    )


    model_path = os.path.join(
        pipeline_params.general.dir,
        pipeline_params.general.model_path,
    )
    logger.info(f'saving model to {model_path}')
    path_to_model = serialize_model(
        model, model_path)

    logger.info('evaluate model on validation dataset')
    metrics = evaluate(X_val_pool, model)
    logger.info(f'validation metrics: {metrics}')


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
    train_pipeline(config)


if __name__ == "__main__":
    main()

