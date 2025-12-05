"""
Guided Rollout MCTS implementation.
"""

import math
from typing import List, Optional
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_dojo.interaction.dojo import DojoTacticTimeoutError

from src.utilities.gym import LeanDojoEnv
from src.agent.mcts.base_mcts import BaseMCTS, Node

# Max depth for a single rollout in Part 1
MAX_ROLLOUT_DEPTH = 30
# Number of tactics to expand from the generator
NUM_TACTICS_TO_EXPAND = 16


class MCTS_GuidedRollout(BaseMCTS):
    """
    Implements Part 1.
    The _simulate method performs a full "guided rollout"
    using the TacticGenerator greedily until the proof is
    finished or max depth is reached.
    """

    def __init__(self, batch_size: int = 16, *args, **kwargs):
        super().__init__(batch_size=batch_size, *args, **kwargs)

    def _ucb1(self, node: Node) -> float:
        """Calculates the UCB1 score for a node."""
        # Virtual loss increases visit count to discourage selection
        visit_count = node.visit_count + self._get_virtual_loss(node)

        if visit_count == 0:
            return float("inf")

        if (
            node.parent is None
        ):  # Should not happen for nodes in _select, but as a safeguard.
            return node.value()

        # Exploitation term
        exploitation = node.value()

        # Exploration term
        # Use parent's visit count.
        # Note: We don't add virtual loss to parent's visit count here,
        # effectively reducing exploration for siblings of nodes being visited.
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.parent.visit_count) / visit_count
        )

        return exploitation + exploration

    def _get_best_child(self, node: Node) -> Node:
        """Selects the best child based on the UCB1 score."""
        return max(node.children, key=self._ucb1)

    def _expand(self, node: Node) -> Node:
        """
        Phase 2: Expansion
        If the node hasn't been expanded yet, generate tactics,
        pick one untried tactic, and create a new child node for it.
        """
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        if node.untried_actions is None:
            # First visit to this node: generate tactics
            state_str = node.state.pp

            node.untried_actions = self.transformer.generate_tactics(
                state_str, n=NUM_TACTICS_TO_EXPAND
            )
            node.untried_actions.reverse()

        # Pop one untried action
        tactic = ""
        if node.untried_actions:
            tactic = node.untried_actions.pop()

        # Run the tactic in the environment to get the next state
        # This is fast as it doesn't modify the main env, just computes the next state
        try:
            next_state_or_result = self.env.dojo.run_tac(node.state, tactic)
        except DojoTacticTimeoutError:
            logger.warning(f"Tactic timed out: {tactic[:100]}")
            # Treat timeout as an error state
            next_state_or_result = LeanError(error="Tactic execution timed out")
        except Exception as e:
            logger.warning(f"Error running tactic '{tactic[:100]}': {e}")
            next_state_or_result = LeanError(error=f"Exception: {str(e)}")

        # Create the new child node
        child = Node(next_state_or_result, parent=node, action=tactic)
        node.children.append(child)
        self.node_count += 1

        return child

    def _expand_batch(self, nodes: List[Node]) -> List[Node]:
        # 1. Generate tactics for all nodes
        states = []
        nodes_to_generate = []
        indices_to_generate = []

        for i, node in enumerate(nodes):
            if node.untried_actions is None:
                if isinstance(node.state, TacticState):
                    states.append(node.state.pp)
                    nodes_to_generate.append(node)
                    indices_to_generate.append(i)
                else:
                    # Should not happen given _select logic
                    node.untried_actions = []

        if states:
            batch_tactics = self.transformer.generate_tactics_batch(
                states, n=NUM_TACTICS_TO_EXPAND
            )
            for i, tactics in enumerate(batch_tactics):
                node = nodes_to_generate[i]
                node.untried_actions = tactics
                node.untried_actions.reverse()

        # 2. Pick one tactic for each node
        tactics_to_run = []
        for node in nodes:
            tactic = ""
            if node.untried_actions:
                tactic = node.untried_actions.pop()
            tactics_to_run.append(tactic)

        # 3. Run tactics sequentially (single env)
        children = []
        for node, tactic in zip(nodes, tactics_to_run):
            if not tactic:
                child = Node(
                    LeanError(error="No tactics generated"), parent=node, action=tactic
                )
            else:
                try:
                    next_state = self.env.dojo.run_tac(node.state, tactic)  # type: ignore
                except Exception as e:
                    next_state = LeanError(error=str(e))
                child = Node(next_state, parent=node, action=tactic)
            children.append(child)

        self.node_count += len(children)
        for i, node in enumerate(nodes):
            node.children.append(children[i])

        return children

    def _simulate(self, node: Node, env: Optional[LeanDojoEnv] = None) -> float:
        """
        Phase 3: Simulation (Guided Rollout)
        """
        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0  # Should not happen if checks are correct

        current_state: TacticState = node.state

        # Use provided env or fallback to self.env
        sim_env = env if env else self.env

        for _ in range(MAX_ROLLOUT_DEPTH):
            state_str = current_state.pp

            # Get a single greedy tactic
            tactic = self.transformer.generate_tactics(state_str, n=1)[0]

            # Run the tactic with timeout handling
            try:
                result = sim_env.dojo.run_tac(current_state, tactic)
            except DojoTacticTimeoutError:
                logger.warning(f"Tactic timed out during simulation: {tactic[:100]}")
                return -1.0  # Penalize timeouts
            except Exception as e:
                logger.warning(
                    f"Error during simulation with tactic '{tactic[:100]}': {e}"
                )
                return -1.0

            # Check result
            if isinstance(result, ProofFinished):
                return 1.0
            if isinstance(result, (LeanError, ProofGivenUp)):
                return -1.0  # Penalize errors

            if not isinstance(result, TacticState):
                return -1.0  # Should not happen

            current_state = result  # Continue rollout

        return 0.0  # Reached max depth, count as a draw/timeout

    def _simulate_batch(self, nodes: List[Node]) -> List[float]:
        return [self._simulate(node) for node in nodes]
