"""Generate quasi random data to simulate batch airflow job run"""
import os

import click
from sklearn.datasets import load_diabetes


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False, header=True)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False, header=False)

if __name__ == '__main__':
    download()
