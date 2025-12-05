"""
PyTorch Datasets for training the policy and value heads.
"""

from typing import List, Dict, Any
from torch.utils.data import Dataset


class ValueHeadDataset(Dataset):
    """Dataset for state -> value_target."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        return {
            "state": item["state"],
            "value_target": item["value_target"],
        }


class PolicyHeadDataset(Dataset):
    """Dataset for (state, premises) -> tactic_target."""

    def __init__(self, data: Dict[Any, Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]
