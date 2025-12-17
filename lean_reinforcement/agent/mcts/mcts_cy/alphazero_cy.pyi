from typing import Any, List, Optional, Tuple
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy import BaseMCTS, Node, Edge

class MCTS_AlphaZero(BaseMCTS):
    value_head: Any
    def __init__(
        self,
        value_head: Any,
        env: Any,
        transformer: Any,
        exploration_weight: float = ...,
        max_tree_nodes: int = ...,
        batch_size: int = ...,
        num_tactics_to_expand: int = ...,
        max_rollout_depth: int = ...,
        max_time: float = ...,
        **kwargs: Any,
    ) -> None: ...
    def _puct_score(self, parent: Node, edge: Edge) -> float: ...
    def _get_best_edge(self, node: Node) -> Edge: ...
    def _expand(self, node: Node) -> Tuple[Node, Optional[Edge]]: ...
    def _expand_batch(self, nodes: List[Node]) -> List[Tuple[Node, Optional[Edge]]]: ...
    def _simulate(self, node: Node) -> float: ...
    def _simulate_batch(self, nodes: List[Node]) -> List[float]: ...
