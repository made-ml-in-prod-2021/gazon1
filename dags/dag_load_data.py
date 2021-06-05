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

with DAG(
    "load_data",
    default_args=default_args,
    description="Load data for model training and predictons",
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
) as dag:
    task_load = DockerOperator(
        image="maxdrobin/airflow-download",
        command="/data/raw/{{ ds }}",
        task_id="docker-airflow-download",
        **docker_kwargs,
    )
