from typing import TypedDict, List, Dict, Tuple, Union, Any, Optional
import torch


class TheoremData(TypedDict):
    url: Optional[str]
    commit: Optional[str]
    file_path: Optional[str]
    full_name: Optional[str]
    start: Tuple[int, int]


class TrainingDataPoint(TypedDict, total=False):
    type: str
    state: str
    step: int
    mcts_value: float
    visit_count: int
    visit_distribution: Dict[str, float]
    value_target: float
    final_reward: float


Metrics = TypedDict(
    "Metrics",
    {
        "proof_search/success": bool,
        "proof_search/steps": int,
        "proof_search/time": float,
    },
)


class WorkerResult(TypedDict, total=False):
    metrics: Metrics
    data: List[TrainingDataPoint]


# Inference Server Types
TacticGenPayload = Tuple[str, int]
TacticGenBatchPayload = Tuple[List[str], int]
ValuePredPayload = Tuple[str]
ValuePredBatchPayload = Tuple[List[str]]
FeaturePayload = Tuple[torch.Tensor]
FeatureBatchPayload = Tuple[torch.Tensor]
EncodeStatesPayload = Tuple[List[str]]

InferencePayload = Union[
    TacticGenPayload,
    TacticGenBatchPayload,
    ValuePredPayload,
    ValuePredBatchPayload,
    FeaturePayload,
    FeatureBatchPayload,
    EncodeStatesPayload,
    Tuple[Any, ...],  # Fallback for other types if any
]

InferenceRequest = Tuple[int, str, InferencePayload]

InferenceResult = Union[
    List[str],
    List[Tuple[str, float]],
    List[float],
    float,
    torch.Tensor,
    List[List[str]],
    List[List[Tuple[str, float]]],
    List[List[float]],
    List[Any],  # Fallback
    None,
]


class MCTSOptions(TypedDict, total=False):
    exploration_weight: float
    max_tree_nodes: int
    batch_size: int
    num_tactics_to_expand: int
    max_rollout_depth: int
    max_time: float
    value_head: Any  # Avoid circular import
