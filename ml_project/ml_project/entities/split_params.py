from dataclasses import dataclass, field


@dataclass()
class SplitParams:
    val_size: float
    random_state: int
    dataset_processed_dir: str
    dataset_processed_train: str
    dataset_processed_val: str
    dataset_processed_test: str
