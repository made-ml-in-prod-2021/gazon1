import os

import pytest
from airflow.models import DagBag


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)


@pytest.fixture()
def dag_bag():
    return DagBag(os.path.join(BASE_DIR, "dags"), include_examples=False)


def test_dags_are_loaded(dag_bag):
    assert not dag_bag.import_errors, (
        f"Not all dags loaded correctly: {dag_bag.import_errors}"
    )

    dag_list = ['dag_load_data', 'dag_predict', 'dag_train']

    assert all(
        [dag_name in dag_bag.dags for dag_name in dag_list]
    ), f"Dag must be {dag_list}"


@pytest.mark.parametrize(
    ["dag_name", "task_list"],
    [
        pytest.param("load_dataset", [
            'docker-airflow-download',
        ]),
        pytest.param("train_model", [
            'check-data',
            'check-target',
            'docker-airflow-split-raw-dataset',
            'docker-airflow-fit-transformer',
            'docker-airflow-transformer',
            'docker-airflow-fit-model',
            'docker-airflow-validate-model',
        ]),
        pytest.param("predict", [
            'check-data',
            'check-transformer',
            'check-model',
            'docker-airflow-transform-data',
            'docker-airflow-predict',
        ]),
    ]
)
def test_dag_contains_all_appropriate_tasks(
    dag_name, task_list, dag_bag
):
    dag = dag_bag.dags.get(dag_name)
    exp_task_list_len = len(task_list)
    given_task_list_len = len(dag.tasks)
    assert given_task_list_len == exp_task_list_len, (
        f"Wrong number of tasks: {given_task_list_len} instead of \
        {exp_task_list_len}"
    )
    for i in range(exp_task_list_len):
        assert dag.tasks[i].task_id == task_list[i], (
            f"At position #{i} must be task {task_list[i]} "
            f"but task given: {dag.tasks[i].task_id}"
        )
