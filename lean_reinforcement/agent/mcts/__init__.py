import os
from typing import TYPE_CHECKING


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


if TYPE_CHECKING:
    from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node
    from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
    from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero
else:
    disable_cython = _env_flag_enabled("LEAN_RL_DISABLE_CYTHON_MCTS")
    if disable_cython:
        from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node
        from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
        from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero
    else:
        try:
            from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy import (
                BaseMCTS,
                Node,
            )
            from lean_reinforcement.agent.mcts.mcts_cy.guidedrollout_cy import (
                MCTS_GuidedRollout,
            )
            from lean_reinforcement.agent.mcts.mcts_cy.alphazero_cy import (
                MCTS_AlphaZero,
            )
        except ImportError:
            from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node
            from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
            from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero

__all__ = ["BaseMCTS", "Node", "MCTS_GuidedRollout", "MCTS_AlphaZero"]
