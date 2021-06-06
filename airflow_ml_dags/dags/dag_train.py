from datetime import timedelta
import os
import random
import sys

from airflow import DAG
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.docker_operator import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'Drobin',
    'retries': 2,
    'enable_xcom_pickling': True,
    'email_on_failure': True,
    'email_on_retry': True,
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
    "Train_model",
    default_args=default_args,
    description="Prepare data and train a model on it",
    schedule_interval="0 2 * * 0",
    start_date=days_ago(7),
) as dag:
    check_data = FileSensor(
        task_id="check-data",
        filepath=f"{DOCKER_DATA_DIR}/raw/{EXECUTION_DATE}/data.csv",
        poke_interval=10,
        retries=2,
    )
    check_target = FileSensor(
        task_id="check-target",
        filepath=f"{DOCKER_DATA_DIR}/raw/{EXECUTION_DATE}/target.csv",
        poke_interval=10,
        retries=2,
    )

    raw_data_split_command = \
        f"--input-dir='/data/raw/{EXECUTION_DATE}' " \
        f"--output-dir='/data/raw-split/{EXECUTION_DATE}' "
    task_split_raw_data = DockerOperator(
        image="maxdrobin/airflow-split-dataset",
        command=raw_data_split_command,
        task_id="docker-airflow-split-raw-dataset",
        **docker_kwargs,
    )

    fit_transform_command = \
        " fit_and_save_transformer " \
        f"--input-dir='/data/raw-split/{EXECUTION_DATE}/train' " \
        f"--output-dir='/data/transformed-split/{EXECUTION_DATE}/train' " \
        "--transformer-dir='/data/models' "
    task_fit_transform = DockerOperator(
        image="maxdrobin/airflow-transform",
        command=fit_transform_command,
        task_id="docker-airflow-fit-transformer",
        **docker_kwargs,
    )

    transform_command = \
        " transform " \
        f"--input-dir='/data/raw-split/{EXECUTION_DATE}/val' " \
        f"--output-dir='/data/transformed-split/{EXECUTION_DATE}/val' " \
        "--transformer-dir='/data/models' "
    task_transform = DockerOperator(
        image="maxdrobin/airflow-transform",
        command=transform_command,
        task_id="docker-airflow-transformer",
        **docker_kwargs,
    )

    fit_command = \
        " train " \
        f"--input-dir-train='/data/transformed-split/{EXECUTION_DATE}/train' " \
        f"--input-dir-val='/data/transformed-split/{EXECUTION_DATE}/val' " \
        f"--model-dir='/data/models' "
    task_fit_model = DockerOperator(
        image="maxdrobin/airflow-model",
        command=fit_command,
        task_id="docker-airflow-fit-model",
        **docker_kwargs,
    )

    validate_command = \
        " evaluate " \
        f"--input-dir-train='/data/transformed-split/{EXECUTION_DATE}/train' " \
        f"--input-dir-val='/data/transformed-split/{EXECUTION_DATE}/val' " \
        f"--output-dir='/data/metrics' " \
        f"--model-dir='/data/models' "
    task_validate_model = DockerOperator(
        image="maxdrobin/airflow-model",
        command=validate_command,
        task_id="docker-airflow-validate-model",
        **docker_kwargs,
    )

    [check_target, check_data] >> task_split_raw_data >> \
        task_fit_transform >> task_transform >> \
        task_fit_model >> task_validate_model
