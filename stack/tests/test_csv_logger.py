"""CsvLogger: fixed columns; fresh runs start over, resumed runs drop rows past the checkpoint."""

import csv

from stack.training.csv_logger import CsvLogger

COLUMNS = ["timesteps", "mean_reward"]


def _read(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_writes_only_the_configured_columns(tmp_path) -> None:
    path = tmp_path / "logs" / "evaluations.csv"
    logger = CsvLogger(path, COLUMNS)
    logger.log({"timesteps": 10, "mean_reward": -1.5, "critic_loss": 0.3})

    assert _read(path) == [{"timesteps": "10", "mean_reward": "-1.5"}]
    assert logger.n_rows == 1


def test_fresh_run_starts_a_new_file(tmp_path) -> None:
    path = tmp_path / "rollouts.csv"
    CsvLogger(path, COLUMNS).log({"timesteps": 1, "mean_reward": 0.5})

    CsvLogger(path, COLUMNS).log({"timesteps": 1, "mean_reward": 0.25})

    assert _read(path) == [{"timesteps": "1", "mean_reward": "0.25"}]


def test_resume_keeps_rows_up_to_the_checkpoint(tmp_path) -> None:
    path = tmp_path / "evaluations.csv"
    logger = CsvLogger(path, COLUMNS)
    for step in (10, 20, 30):
        logger.log({"timesteps": step, "mean_reward": -step})

    resumed = CsvLogger(path, COLUMNS, resume_step=20)
    assert resumed.n_rows == 2

    resumed.log({"timesteps": 30, "mean_reward": -31})
    assert [row["mean_reward"] for row in _read(path)] == ["-10", "-20", "-31"]


def test_resume_rewrites_an_older_wider_file_to_the_current_columns(tmp_path) -> None:
    path = tmp_path / "evaluations.csv"
    CsvLogger(path, ["timesteps", "mean_reward", "episode_rewards"]).log(
        {"timesteps": 5, "mean_reward": 1.0, "episode_rewards": "[1.0]"}
    )

    CsvLogger(path, COLUMNS, resume_step=5)

    assert _read(path) == [{"timesteps": "5", "mean_reward": "1.0"}]


def test_resume_without_existing_file_starts_empty(tmp_path) -> None:
    logger = CsvLogger(tmp_path / "missing.csv", COLUMNS, resume_step=100)
    assert logger.n_rows == 0
    logger.log({"timesteps": 101, "mean_reward": 2})
    assert _read(tmp_path / "missing.csv") == [{"timesteps": "101", "mean_reward": "2"}]
