from datetime import timedelta
import os
import random
import sys

from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'Drobin'
    , 'retries': 2
    , 'enable_xcom_pickling': True,
}

docker_kwargs = {
    'network_mode': "bridge",
    'volumes': ["/home/max/MADE/ml-prod/gazon1/data:/data"]
}

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
        "--input-dir='/data/raw-split/{EXECUTION_DATE}/train' " \
        "--output-dir='/data/transformed-split/{EXECUTION_DATE}/train' " \
        "--transformer-dir='/data/models' "
    task_fit_transform = DockerOperator(
        image="maxdrobin/airflow-transform",
        command=fit_transform_command,
        task_id="docker-airflow-fit-transformer",
        **docker_kwargs,
    )

    transform_command = \
        " transform " \
        "--input-dir='/data/raw-split/{EXECUTION_DATE}/val' " \
        "--output-dir='/data/transformed-split/{EXECUTION_DATE}/val' " \
        "--transformer-dir='/data/models' "
    task_ftransform = DockerOperator(
        image="maxdrobin/airflow-transform",
        command=fit_transform_command,
        task_id="docker-airflow-transformer",
        **docker_kwargs,
    )

    fit_command = \
        " train " \
        f"--input-dir-train='/data/transformed-split/{EXECUTION_DATE}/train' " \
        f"--input-dir-val='/data/transformed-split/{EXECUTION_DATE}/val' " \
        f"--output-dir='/data/models' "
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

    task_split_raw_data >> task_fit_transform >> \
        task_ftransform >> task_fit_model >> task_validate_model
