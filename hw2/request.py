"""Make POST requests to REST classification service"""
import os

import click
import requests

from app.utils.utils import (
    generate_example_from_real_dataset,
)
from app.main import DATA_DIR

SERVICE_HOST = os.environ.get("HOST", default="127.0.0.1")
SERVICE_PORT = os.environ.get("PORT", default=8000)
REAL_DATASET_TMP_FILEPATH = os.path.join(DATA_DIR, "data.csv")


@click.command()
@click.option("--count", default=1, help="number of API requests")
def make_request(count: int):
    """
    Send requests for REST api service to get predictions
    Params:
        count : int - number of requests
    """
    get_data = lambda: generate_example_from_real_dataset(
        REAL_DATASET_TMP_FILEPATH
    )

    for _ in range(count):
        data = get_data()
        response = requests.post(
            url=f"http://{SERVICE_HOST}:{SERVICE_PORT}/predict",
            json=data
        )
        if response.status_code == 200:
            click.echo(f"Request:\t {data}")
            click.echo(f"Response:\t {response.json()}")
        else:
            click.echo(f"ERROR {response.status_code}: {response.text}")

        click.echo("---")


if __name__ == "__main__":
    make_request()
