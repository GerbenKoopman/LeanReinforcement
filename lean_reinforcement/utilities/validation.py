from typing import Any, List, TypeGuard, Tuple
import torch
from lean_reinforcement.utilities.types import (
    TheoremData,
    TacticGenPayload,
    TacticGenBatchPayload,
    TrainingDataPoint,
    ValuePredPayload,
    ValuePredBatchPayload,
    FeaturePayload,
    FeatureBatchPayload,
    EncodeStatesPayload,
    InferencePayload,
)


def is_theorem_data(data: Any) -> TypeGuard[TheoremData]:
    return (
        isinstance(data, dict)
        and (data.get("url") is None or isinstance(data.get("url"), str))
        and (data.get("commit") is None or isinstance(data.get("commit"), str))
        and (data.get("file_path") is None or isinstance(data.get("file_path"), str))
        and (data.get("full_name") is None or isinstance(data.get("full_name"), str))
        and isinstance(data.get("start"), (list, tuple))
        and len(data["start"]) == 2
        and all(isinstance(x, int) for x in data["start"])
    )


def is_list_of_theorem_data(data: Any) -> TypeGuard[List[TheoremData]]:
    return isinstance(data, list) and all(is_theorem_data(item) for item in data)


def is_tactic_gen_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[TacticGenPayload]]:
    if not batch:
        return True
    item = batch[0]
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], int)
    )


def is_tactic_gen_batch_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[TacticGenBatchPayload]]:
    if not batch:
        return True
    item = batch[0]
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], list)
        and all(isinstance(x, str) for x in item[0])
        and isinstance(item[1], int)
    )


def is_value_pred_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[ValuePredPayload]]:
    if not batch:
        return True
    item = batch[0]
    return isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], str)


def is_value_pred_batch_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[ValuePredBatchPayload]]:
    if not batch:
        return True
    item = batch[0]
    return (
        isinstance(item, tuple)
        and len(item) == 1
        and isinstance(item[0], list)
        and all(isinstance(x, str) for x in item[0])
    )


def is_feature_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[FeaturePayload]]:
    if not batch:
        return True
    item = batch[0]
    return (
        isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], torch.Tensor)
    )


def is_feature_batch_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[FeatureBatchPayload]]:
    if not batch:
        return True
    item = batch[0]
    return (
        isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], torch.Tensor)
    )


def is_encode_states_payload_list(
    batch: List[InferencePayload],
) -> TypeGuard[List[EncodeStatesPayload]]:
    if not batch:
        return True
    item = batch[0]
    return (
        isinstance(item, tuple)
        and len(item) == 1
        and isinstance(item[0], list)
        and all(isinstance(x, str) for x in item[0])
    )


def validate_inference_result_list_str(result: Any) -> List[str]:
    if isinstance(result, list) and all(isinstance(x, str) for x in result):
        return result
    raise TypeError(f"Expected List[str], got {type(result)}")


def validate_inference_result_list_tuple_str_float(
    result: Any,
) -> List[Tuple[str, float]]:
    if isinstance(result, list) and all(
        isinstance(x, (list, tuple))
        and len(x) == 2
        and isinstance(x[0], str)
        and isinstance(x[1], float)
        for x in result
    ):
        return result
    raise TypeError(f"Expected List[Tuple[str, float]], got {type(result)}")


def validate_inference_result_list_list_str(result: Any) -> List[List[str]]:
    if isinstance(result, list) and all(
        isinstance(x, list) and all(isinstance(y, str) for y in x) for x in result
    ):
        return result
    raise TypeError(f"Expected List[List[str]], got {type(result)}")


def validate_inference_result_list_list_tuple_str_float(
    result: Any,
) -> List[List[Tuple[str, float]]]:
    if isinstance(result, list) and all(
        isinstance(x, list)
        and all(
            isinstance(y, (list, tuple))
            and len(y) == 2
            and isinstance(y[0], str)
            and isinstance(y[1], float)
            for y in x
        )
        for x in result
    ):
        return result
    raise TypeError(f"Expected List[List[Tuple[str, float]]], got {type(result)}")


def validate_float(result: Any) -> float:
    if isinstance(result, float):
        return result
    raise TypeError(f"Expected float, got {type(result)}")


def validate_list_float(result: Any) -> List[float]:
    if isinstance(result, list) and all(isinstance(x, float) for x in result):
        return result
    raise TypeError(f"Expected List[float], got {type(result)}")


def validate_torch_tensor(result: Any) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    raise TypeError(f"Expected torch.Tensor, got {type(result)}")


def validate_list_list_float(result: Any) -> List[List[float]]:
    if isinstance(result, list) and all(
        isinstance(x, list) and all(isinstance(y, float) for y in x) for x in result
    ):
        return result
    raise TypeError(f"Expected List[List[float]], got {type(result)}")


def is_training_data_point(data: Any) -> TypeGuard[TrainingDataPoint]:
    return isinstance(data, dict)


def is_list_of_training_data_point(data: Any) -> TypeGuard[List[TrainingDataPoint]]:
    return isinstance(data, list) and all(is_training_data_point(x) for x in data)
