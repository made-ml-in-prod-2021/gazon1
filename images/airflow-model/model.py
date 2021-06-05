import os
import json
import pickle
import logging
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, Pool


logger = logging.getLogger(__name__)

TRAIN_PARAMS = {
    'bagging_temperature': 1
    'bootstrap_type': 'Bayesian'
    'l2_leaf_reg': 1
    'learning_rate': 0.6
    'loss_function': 'Logloss'
    'max_depth': 5
    'n_estimators': 300
    'random_strength': 0.5
    'verbose': True
    'use_best_model': True
}

def serialize_model(model, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

def deserialize_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def train_model(
    X_train: Pool, X_val: Pool,
    train_params=TRAIN_PARAMS: dict,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**train_params)
    model.fit(X_train, eval_set=X_val)
    return model

def evaluate(
    X_val: Pool,
    model: CatBoostClassifier,
    metrics: List[str]=['Precision', 'Recall',
                        'F1', 'AUC:type=Classic', 'Accuracy']
):
    return model.eval_metrics(
        X_val,
        metrics=metrics,
        eval_period=model.tree_count_
    )


@click.command("train")
@click.option("--input-dir-train", type=click.Path(exists=True))
@click.option("--input-dir-val", type=click.Path(exists=True))
@click.option("--model-dir", type=click.Path())
def train_pipeline(input_dir_train: str, input_dir_val: str,
                   model_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir_train, "data.csv"))
    y_train = pd.read_csv(os.path.join(input_dir_train, "target.csv"))

    X_val = pd.read_csv(os.path.join(input_dir_val, "data.csv"))
    y_val = pd.read_csv(os.path.join(input_dir_val, "target.csv"))

    X_train_pool = Pool(X_train, y_train)
    X_val_pool = Pool(X_val, y_val)

    logger.info('train model')
    model = train_model(
        X_train_pool,
        X_val_pool,
    )

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir, 'model.cbm',
    )
    logger.info(f'saving model to {model_path}')
    path_to_model = serialize_model(
        model, model_path)


@click.command("evaluate")
@click.option("--model-dir", type=click.Path(exists=True))
@click.option("--input-dir-train", type=click.Path(exists=True))
@click.option("--input-dir-val", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
def evaluate_pipeline(model_dir: str, input_dir_train: str,
                       input_dir_val: str, output_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir_train, "data.csv"))
    y_train = pd.read_csv(os.path.join(input_dir_train, "target.csv"))

    X_val = pd.read_csv(os.path.join(input_dir_val, "data.csv"))
    y_val = pd.read_csv(os.path.join(input_dir_val, "target.csv"))

    X_train_pool = Pool(X_train, y_train)
    X_val_pool = Pool(X_val, y_val)

    logger.info('evaluate model on validation dataset')
    metrics = evaluate(X_val_pool, model)
    logger.info(f'validation metrics: {metrics}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), "w") as fout:
        json.dump(metrics , fout)


@click.command("predict")
@click.option("--input-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--model-dir", type=click.Path(exists=True))
def evaluate_pipeline(model_dir: str, input_dir_train: str,
                       input_dir_val: str, output_dir: str):
    X_test = pd.read_csv(os.path.join(input_dir, "data.csv"))
    X_test['predictions'] = model.predict(X_test)
    X_test.to_csv(os.path.join(
        output_dir, 'predictions.csv'
    ), index=False)


@click.group()
def cli():
  pass


cli.add_command(train_pipeline)
cli.add_command(evaluate_pipeline)
cli.add_command(predict_pipeline)

if __name__ == "__main__":
    cli()
