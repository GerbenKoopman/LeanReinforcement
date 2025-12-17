"""
PyTorch Datasets for training the policy and value heads.
"""

from typing import List, TypedDict
from torch.utils.data import Dataset
from lean_reinforcement.utilities.types import TrainingDataPoint


class ValueData(TypedDict):
    state: str
    value_target: float


class ValueHeadDataset(Dataset):
    """Dataset for state -> value_target."""

    def __init__(self, data: List[TrainingDataPoint]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> ValueData:
        item = self.data[idx]
        return {
            "state": item.get("state", ""),
            "value_target": item.get("value_target", 0.0),
        }
