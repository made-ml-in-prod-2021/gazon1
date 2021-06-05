from datetime import timedelta
import os
import sys

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'Drobin'
    , 'retries': 2
    , 'enable_xcom_pickling': True,
}

docker_kwargs = {
    'network_mode': "bridge",
    'volumes': ["'/home/max/MADE/ml-prod/gazon1/data':/data"]
}

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

    # check_data = FileSensor(
    #     task_id="check-data-exists",
    #     filepath=f"{AIRFLOW_BASE_DIR}/data/raw/{EXECUTION_DATE}/data.csv",
    #     poke_interval=30,
    #     retries=100,
    # )
    # check_transformer = FileSensor(
    #     task_id="check-transformer-exists",
    #     filepath=f"{AIRFLOW_BASE_DIR}/data/models/transformer.pickle",
    #     poke_interval=30,
    #     retries=100,
    # )
    # check_model = FileSensor(
    #     task_id="check-model-exists",
    #     filepath=f"{AIRFLOW_BASE_DIR}/data/models/model.pickle",
    #     poke_interval=30,
    #     retries=100,
    # )

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

    # task_start >> [
        # check_data, check_transformer, check_model
    # ] >> task_transform_data >> task_predict
    task_transform_data >> task_predict
