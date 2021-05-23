"""Make POST requests to REST classification service"""
import os

import click
import requests

from app.utils.utils import (
    generate_example_from_real_dataset,
    # generate_example_from_synthetic_dataset
)
from app.main import DATA_DIR

SERVICE_HOST = os.environ.get("HOST", default="127.0.0.1")
SERVICE_PORT = os.environ.get("PORT", default=8000)
REAL_DATASET_TMP_FILEPATH = os.path.join(DATA_DIR, "data.csv")


@click.command()
@click.option("--count", default=1, help="number of API requests")
@click.option("--random-data", is_flag=True)
def make_request(count: int, random_data: bool):
    """Provide POST requests to classification REST service.
    Prints requested data and server response to command line.
    Params:
        count : int - number of requests
        random_data : bool - whether to use randomly generated data
    """
    # if random_data:
    #     get_data = generate_example_from_synthetic_dataset
    # else:
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
