import os
import shutil
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .utils.utils import (
    setup_logger,
    Model,
)

from .entities import (
    OutputModelData,
    InputModelData,
)

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir
    )
)
MODELS_DIR = os.path.join(BASE_DIR, "app", "models")
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

setup_logger()
logger = logging.getLogger()

app = FastAPI()

model = Model(
    transformer_path = os.path.join(MODELS_DIR, "transformer.pkl"),
    model_path = os.path.join(MODELS_DIR, "model.cbm"),
)


@app.on_event("startup")
def load_model():
    """
    load transformer and model and wrap all in pipeline
    with global name `model`
    """
    global model
    model.load()

@app.get("/")
def root():
    return "HW2: REST services"

@app.get("/health")
def check_health() -> bool:
    """Check: pretrained classifier and transformer loaded correctly"""
    global model
    return (model.model is not None or model.transformer is None)


@app.post("/predict")
def make_prediction(data: InputModelData) -> OutputModelData:
    global model
    if not check_health():
        logger.error("Model is not loaded")
        raise HTTPException(
            status_code=500,
            detail="Model should be loaded for making predictions"
        )
    pred_prob = model.predict(
        data=data.to_pandas(),
    )
    return {"predicted_probability": pred_prob}
