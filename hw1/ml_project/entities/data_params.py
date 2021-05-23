from dataclasses import dataclass


@dataclass()
class DataParams:
    dataset_raw_dir: str
    dataset_raw_file: str
    dataset_raw_train: str
    dataset_raw_val: str
    dataset_raw_test: str
