from dataclasses import dataclass, field


@dataclass()
class TrainParams:
    bagging_temperature: int = field(default=5)
    bootstrap_type: str = field(default="Bayesian")
    l2_leaf_reg: float = field(default=3)
    learning_rate: float = field(default=0.2)
    loss_function: str = field(default="YetiRank")
    max_depth: int = field(default=6)
    n_estimators: int = field(default=150)
    random_strength: float = field(default=0.2)
    verbose: bool = field(default=True)
    use_best_model: bool = field(default=True)
