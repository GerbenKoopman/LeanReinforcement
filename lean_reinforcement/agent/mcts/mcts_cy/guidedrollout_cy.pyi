from typing import Any, List, Optional, Tuple
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy import BaseMCTS, Node, Edge
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol

class MCTS_GuidedRollout(BaseMCTS):
    def __init__(
        self,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
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
