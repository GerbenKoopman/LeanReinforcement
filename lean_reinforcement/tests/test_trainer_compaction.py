import json
import queue
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import torch.multiprocessing as mp

from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.training.progress import (
    ProgressStats,
    LiveProgressDisplay,
    PlainProgressDisplay,
    NullProgressDisplay,
)
from lean_reinforcement.training.inference_server import InferenceServer
from lean_reinforcement.training.trainer import Trainer


class _DummyInferenceServer:
    def process_requests(self) -> bool:
        return False


class _DummyProgressStats:
    def __init__(self) -> None:
        self.alive_workers = 0

    def record_theorem(self, **kwargs) -> None:
        _ = kwargs


class _DummyProgressDisplay:
    def refresh(self) -> None:
        return


def _make_minimal_trainer(tmp_path: Path) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.checkpoint_dir = tmp_path
    trainer.config = cast(
        TrainingConfig,
        SimpleNamespace(
            num_workers=1,
            proof_timeout=10.0,
            inference_timeout=10.0,
            use_wandb=False,
            debugging=False,
            experience_replay_max_epochs=None,
        ),
    )
    trainer.workers = []
    trainer.theorem_queue = cast(mp.Queue, queue.Queue())
    trainer.request_queue = cast(mp.Queue, queue.Queue())
    trainer.response_queues = []
    trainer.result_queue = cast(mp.Queue, queue.Queue())
    trainer.progress_stats = cast(ProgressStats, _DummyProgressStats())
    trainer.progress_display = cast(
        LiveProgressDisplay | PlainProgressDisplay | NullProgressDisplay,
        _DummyProgressDisplay(),
    )
    return trainer


def test_collect_data_compacts_rich_samples(tmp_path, monkeypatch) -> None:
    trainer = _make_minimal_trainer(tmp_path)
    monkeypatch.setattr(
        "lean_reinforcement.training.trainer.get_available_memory_gb",
        lambda: 999.0,
    )

    rich_value_sample = {
        "type": "value",
        "state": "goal_state",
        "value_target": 1.0,
        "mcts_value": 0.25,
        "step": 3,
        "visit_count": 42,
        "visit_distribution": {"exact": 1.0},
    }

    trainer.result_queue.put(
        {
            "theorem_name": "thm",
            "metrics": {
                "proof_search/success": True,
                "proof_search/steps": 1,
                "proof_search/time": 0.1,
            },
            "data": [rich_value_sample],
        }
    )

    data, metrics = trainer._collect_data(
        theorems_to_process=["thm"],
        inference_server=cast(InferenceServer, _DummyInferenceServer()),
        epoch=0,
    )

    assert len(metrics) == 1
    assert data == [
        {
            "type": "value",
            "state": "goal_state",
            "value_target": 1.0,
            "mcts_value": 0.25,
        }
    ]


def test_load_experience_replay_honors_epoch_limit(tmp_path) -> None:
    trainer = _make_minimal_trainer(tmp_path)
    trainer.config.experience_replay_max_epochs = 1

    old_path = tmp_path / "training_data_epoch_1.json"
    new_path = tmp_path / "training_data_epoch_2.json"

    with open(old_path, "w") as f:
        json.dump([{"type": "value", "state": "old_state", "value_target": -1.0}], f)

    with open(new_path, "w") as f:
        json.dump(
            [{"type": "value", "state": "new_state", "value_target": 1.0}],
            f,
        )

    replay = trainer._load_experience_replay_data()
    assert [sample["state"] for sample in replay] == ["new_state"]
