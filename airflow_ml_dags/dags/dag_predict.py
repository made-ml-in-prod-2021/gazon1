from datetime import timedelta
import os
import sys

from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'Drobin',
    'retries': 2,
    'email_on_failure': True,
    'email_on_retry': True,
    'enable_xcom_pickling': True,
    'email': ['drobin.me@yandex.ru'],
}

docker_kwargs = {
    'network_mode': "bridge",
    'volumes': ["/home/max/MADE/ml-prod/gazon1/data:/data"]
}


DOCKER_DATA_DIR = '/opt/airflow/data'

# put current date in variable to use
# one date on every task. If task_load starts at 23:59
# and task_split starts at 00:00 then {{ ds }} in different
# places may return different results
EXECUTION_DATE = '{{ ds }}'


with DAG(
    "predict",
    default_args=default_args,
    description="prepare data and make predictions for it",
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
) as dag:

    check_data = FileSensor(
        task_id="check-data",
        filepath=f"{DOCKER_DATA_DIR}/raw/{EXECUTION_DATE}/data.csv",
        poke_interval=10,
        retries=2,
    )
    check_transformer = FileSensor(
        task_id="check-transformer",
        filepath=f"{DOCKER_DATA_DIR}/models/transformer.pkl",
        poke_interval=10,
        retries=2,
    )
    check_model = FileSensor(
        task_id="check-model",
        filepath=f"{DOCKER_DATA_DIR}/models/model.cbm",
        poke_interval=10,
        retries=2,
    )

    transform_command = \
        " transform " \
        f"--input-dir='/data/raw/{EXECUTION_DATE}' " \
        f"--output-dir='/data/transformed/{EXECUTION_DATE}' " \
        f"--transformer-dir='/data/models' "
    task_transform_data = DockerOperator(
        image="maxdrobin/airflow-transform",
        command=transform_command,
        task_id="docker-airflow-transform-data",
        **docker_kwargs,
    )

    predict_command = \
        " predict " \
        f"--input-dir='/data/transformed/{EXECUTION_DATE}' " \
        f"--output-dir='/data/predictions/{EXECUTION_DATE}' " \
        "--model-dir='/data/models'"
    task_predict = DockerOperator(
        image="maxdrobin/airflow-model",
        command=predict_command,
        task_id="docker-airflow-predict",
        **docker_kwargs,
    )

    [check_data, check_transformer, check_model] >> \
        task_transform_data >> task_predict
