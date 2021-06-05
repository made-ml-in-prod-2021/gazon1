import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

VAL_SIZE = 0.2

@click.command("split_dataset")
@click.option("--input-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
def split_dataset(input_dir: str, output_dir: str, val_size=VAL_SIZE) -> None:
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"), header=None)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=val_size, shuffle=True,
    )

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "train", "data.csv"),
                   header=True, index=False)
    y_train.to_csv(os.path.join(output_dir, "train", "target.csv"),
                   header=True, index=False)

    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "val", "data.csv"),
                   header=True, index=False)
    y_train.to_csv(os.path.join(output_dir, "val", "target.csv"),
                   header=True, index=False)


if __name__ == "__main__":
    split_dataset()
