from dataclasses import dataclass

@dataclass()
class GeneralParams:
    random_state: int
    dir: str
    model_path: str
