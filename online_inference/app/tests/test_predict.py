import pandas as pd
import os
import sys

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
    )
)

sys.path.append(BASE_DIR)

import pytest
from fastapi.testclient import TestClient
from app.utils.utils import (
    Model,
    generate_example_from_real_dataset,
)
from app.entities.input import InputModelData
from app.entities.output import OutputModelData
from app.main import app, MODELS_DIR, DATA_DIR

DATA_PATH = os.path.join(DATA_DIR, "data.csv")


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def data_sample():
    features = {
        'age': 35, 'sex': 0, 'cp': 0,
        'trestbps': 138, 'chol': 183, 'fbs': 0,
        'restecg': 1, 'thalach': 182, 'exang': 0,
        'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
    }
    data = InputModelData(**features).to_pandas()

    model = Model(
        transformer_path = os.path.join(MODELS_DIR, "transformer.pkl"),
        model_path = os.path.join(MODELS_DIR, "model.cbm"),
    )
    model.load()
    preds = model.predict(data)
    return features, preds

def test_base_model_request_column_order():

    features = {
        'age': 45, 'sex': 1, 'cp': 3, 'trestbps': 110,
        'chol': 264, 'fbs': 0, 'restecg': 1, 'thalach': 132,
        'exang': 0, 'oldpeak': 1.2, 'slope': 1, 'ca': 0, 'thal': 3
    }

    data_base_model = InputModelData(**features).to_pandas()
    real_data = generate_example_from_real_dataset(DATA_PATH)
    assert list(real_data.keys()) == data_base_model.columns.tolist(), \
        "column order not correct!"

def test_pred_float_and_between_zero_one():
    features = {
        'age': 35, 'sex': 0, 'cp': 0,
        'trestbps': 138, 'chol': 183, 'fbs': 0,
        'restecg': 1, 'thalach': 182, 'exang': 0,
        'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
    }

    # data = pd.DataFrame.from_dict([features], orient='columns')
    data = InputModelData(**features).to_pandas()

    model = Model(
        transformer_path = os.path.join(MODELS_DIR, "transformer.pkl"),
        model_path = os.path.join(MODELS_DIR, "model.cbm"),
    )
    model.load()
    preds = model.predict(data)
    assert isinstance(preds, float) and 0 <= preds <= 1, \
        "prediction must be probability of type float and between 0 and 1"


def test_predict_returns_accurate_probability(data_sample, client):
    expected_status_code = 200
    expected = {"predicted_probability": data_sample[1]}
    response = client.post("/predict", json=data_sample[0])
    assert response.status_code == expected_status_code
    assert response.json() == expected
