from typing import List, Optional, Dict, Any, Tuple

class Node:
    state: Any
    _pp: Optional[str]
    parents: List[
        Tuple[Node, Optional[str]]
    ]  # DAG structure: list of (parent, action) tuples
    action: Optional[str]
    prior_p: float
    children: List[Node]
    visit_count: int
    max_value: float
    is_terminal: bool
    untried_actions: Optional[List[str]]
    encoder_features: Optional[Any]
    depth: int

    @property
    def parent(self) -> Optional[Node]: ...  # Backward compatibility property
    def __init__(
        self, state: Any, parent: Optional[Node] = ..., action: Optional[str] = ...
    ) -> None: ...
    def value(self) -> float: ...
    def is_fully_expanded(self) -> bool: ...
    def add_parent(self, parent: Node, action: Optional[str] = ...) -> None: ...
    def get_parent(self) -> Optional[Node]: ...

class BaseMCTS:
    env: Any
    transformer: Any
    exploration_weight: float
    max_tree_nodes: int
    batch_size: int
    num_tactics_to_expand: int
    max_rollout_depth: int
    max_time: float
    node_count: int
    virtual_losses: Dict[Node, int]
    seen_states: Dict[str, Node]
    log_search_tree: bool
    q_weight: float
    theorem: Any
    theorem_pos: Any
    root: Node

    def __init__(
        self,
        env: Any,
        transformer: Any,
        config: Any,
        exploration_weight: float = ...,
        max_tree_nodes: int = ...,
        batch_size: int = ...,
        num_tactics_to_expand: int = ...,
        max_rollout_depth: int = ...,
        max_time: float = ...,
        log_search_tree: bool = ...,
        q_weight: float = ...,
        **kwargs: Any,
    ) -> None: ...
    def _get_virtual_loss(self, node: Node) -> int: ...
    def _add_virtual_loss(self, node: Node, loss: int = ...) -> None: ...
    def _remove_virtual_loss(self, node: Node, loss: int = ...) -> None: ...
    def _get_state_key(self, state: Any) -> Optional[str]: ...
    def search(
        self,
        num_iterations: int,
        batch_size: Optional[int] = ...,
        max_time: Optional[float] = ...,
        search_tree_log_dir: Optional[str] = ...,
    ) -> None: ...
    def _select(self, node: Node) -> Node: ...
    def _get_best_child(self, node: Node) -> Node: ...
    def _expand(self, node: Node) -> Node: ...
    def _expand_batch(self, nodes: List[Node]) -> List[Node]: ...
    def _simulate(self, node: Node, env: Any = ...) -> float: ...
    def _simulate_batch(self, nodes: List[Node]) -> List[float]: ...
    def _backpropagate(self, node: Node, reward: float) -> None: ...
    def get_best_action(self) -> Optional[str]: ...
    def extract_proof_path(self) -> Optional[List[str]]: ...
    def move_root(self, action: str) -> None: ...
    def _count_nodes(self, node: Node) -> int: ...
    def _rebuild_seen_states(self, node: Node) -> None: ...
