from typing import List

import catboost


def evaluate(
    X_val: catboost.Pool,
    model: catboost.CatBoostClassifier,
    metrics: List[str]=['Precision', 'Recall',
                        'F1', 'AUC:type=Classic', 'Accuracy']
):
    return model.eval_metrics(
        X_val,
        metrics=metrics,
        eval_period=model.tree_count_
    )
