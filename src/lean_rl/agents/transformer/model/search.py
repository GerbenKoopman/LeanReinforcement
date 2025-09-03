"""
Implements the HierarchicalSearchTree for finding the best proof tactic.
"""

import time
from dataclasses import dataclass
from queue import PriorityQueue
from typing import List, Optional, Union, overload, Literal, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from lean_dojo import TacticState

from .action import HierarchicalAction, format_action_to_string
from .hierarchy import HierarchyLevel, StrategicActions, TacticalFamilies

if TYPE_CHECKING:
    from .agent import HierarchicalTransformerAgent


@dataclass
class SearchNode:
    """Node in the hierarchical search tree."""

    state: TacticState
    parent: Optional["SearchNode"]
    children: List["SearchNode"]
    level: HierarchyLevel
    action: Optional[str]
    value: float
    visits: int
    strategic_action: Optional[str] = None
    tactic_family: Optional[str] = None
    depth: int = 0
    log_prob: float = 0.0

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.value > other.value  # Higher value = higher priority


class HierarchicalSearchTree:
    """
    A hierarchical best-first search tree for finding the best action.

    The tree has three levels:
    1. Strategic: High-level proof strategies (e.g., induction, direct proof).
    2. Tactical: Families of tactics (e.g., rewriting, applying lemmas).
    3. Execution: Specific tactics with parameters.
    """

    def __init__(self, agent: "HierarchicalTransformerAgent", state: TacticState):
        self.agent = agent
        self.root = SearchNode(
            state=state,
            parent=None,
            children=[],
            level=HierarchyLevel.STRATEGIC,
            action=None,
            value=0.0,
            visits=0,
            strategic_action=None,
            depth=0,
            log_prob=0.0,
        )
        self.open_nodes: PriorityQueue[SearchNode] = PriorityQueue()
        self.open_nodes.put(self.root)
        self.nodes_expanded = 0

    @overload
    def search(
        self, max_time: float, beam_width: int, return_log: Literal[True]
    ) -> Tuple[Optional[Tuple[str, SearchNode, HierarchicalAction]], list]: ...

    @overload
    def search(
        self, max_time: float, beam_width: int, return_log: Literal[False]
    ) -> Optional[Tuple[str, SearchNode, HierarchicalAction]]: ...

    @overload
    def search(
        self, max_time: float, beam_width: int
    ) -> Optional[Tuple[str, SearchNode, HierarchicalAction]]: ...

    @overload
    def search(self) -> Optional[Tuple[str, SearchNode, HierarchicalAction]]: ...

    def search(
        self, max_time: float = 60.0, beam_width: int = 16, return_log: bool = False
    ) -> Union[
        Optional[Tuple[str, SearchNode, HierarchicalAction]],
        Tuple[Optional[Tuple[str, SearchNode, HierarchicalAction]], list],
    ]:
        """
        Perform hierarchical best-first search.

        Args:
            max_time: Maximum search time in seconds
            beam_width: Maximum number of nodes to keep in beam
            return_log: Whether to return logging information

        Returns:
            A tuple (best action string, best node, hierarchical action) or None.
            If return_log is True, returns ((action, node, hier_action), log_data).
        """
        start_time = time.time()
        best_action_str: Optional[str] = None
        best_node: Optional[SearchNode] = None
        best_hierarchical_action: Optional[HierarchicalAction] = None

        candidate_nodes: List[tuple[str, SearchNode, HierarchicalAction]] = []

        while (
            not self.open_nodes.empty()
            and time.time() - start_time < max_time
            and self.nodes_expanded < 1000
        ):
            current_node = self.open_nodes.get()

            if current_node.depth >= 2:
                action_obj = self._construct_action_from_node(current_node)
                if action_obj:
                    action_str = format_action_to_string(action_obj)
                    candidate_nodes.append((action_str, current_node, action_obj))
                    # Continue searching for better options within the time limit.

            self._expand_node(current_node)

            if self.open_nodes.qsize() > beam_width:
                self._prune_beam(beam_width)

        if candidate_nodes:
            # Select the best candidate based on value
            best_action_str, best_node, best_hierarchical_action = max(
                candidate_nodes, key=lambda item: item[1].value
            )

        result = (
            (best_action_str, best_node, best_hierarchical_action)
            if best_action_str and best_node and best_hierarchical_action
            else None
        )

        if return_log:
            log_data = self._get_search_log()
            return result, log_data
        else:
            return result

    def _get_search_log(self) -> list:
        """Get top nodes from the search for logging."""
        log_data = []
        nodes = []
        temp_queue = PriorityQueue()

        while not self.open_nodes.empty():
            nodes.append(self.open_nodes.get())

        for node in nodes:
            temp_queue.put(node)
        self.open_nodes = temp_queue

        for node in sorted(nodes, key=lambda x: x.value, reverse=True)[:5]:
            action_obj = self._construct_action_from_node(node)
            action_str = (
                format_action_to_string(action_obj)
                if action_obj
                else f"Incomplete ({node.level.name}, value={node.value:.3f})"
            )
            log_data.append((action_str, node.value))

        return log_data

    def _expand_node(self, node: SearchNode) -> None:
        """Expand a search node by generating child nodes."""
        self.nodes_expanded += 1

        if node.level == HierarchyLevel.STRATEGIC:
            for action in StrategicActions.ALL_ACTIONS[:3]:
                value, log_prob = self._evaluate_strategic_action(node.state, action)
                child = SearchNode(
                    state=node.state,
                    parent=node,
                    children=[],
                    level=HierarchyLevel.TACTICAL,
                    action=action,
                    value=value,
                    visits=0,
                    strategic_action=action,
                    depth=node.depth + 1,
                    log_prob=log_prob,
                )
                node.children.append(child)
                self.open_nodes.put(child)

        elif node.level == HierarchyLevel.TACTICAL:
            for family in TacticalFamilies.ALL_FAMILIES[:3]:
                value, log_prob = self._evaluate_tactical_family(
                    node.state, node.strategic_action or "direct_proof", family
                )
                child = SearchNode(
                    state=node.state,
                    parent=node,
                    children=[],
                    level=HierarchyLevel.EXECUTION,
                    action=family,
                    value=value,
                    visits=0,
                    strategic_action=node.strategic_action,
                    tactic_family=family,
                    depth=node.depth + 1,
                    log_prob=node.log_prob + log_prob,
                )
                node.children.append(child)
                self.open_nodes.put(child)

    def _evaluate_strategic_action(
        self, state: TacticState, action: str
    ) -> tuple[float, float]:
        """Evaluate strategic action using neural network."""
        try:
            encoded_state = self.agent.encode_state(state)
            with torch.no_grad():
                output = self.agent.hierarchical_forward(
                    encoded_state, HierarchyLevel.STRATEGIC
                )
                value = output["value"].item()
                logits = output["policy_logits"]
                log_probs = F.log_softmax(logits, dim=-1)
                action_idx = StrategicActions.ALL_ACTIONS.index(action)
                log_prob = log_probs[0, action_idx].item()
                return value, log_prob
        except (ValueError, IndexError):
            return 0.0, -10.0  # Penalize if action not found
        except Exception:
            return 0.5, 0.0

    def _evaluate_tactical_family(
        self, state: TacticState, strategic_action: str, tactic_family: str
    ) -> tuple[float, float]:
        """Evaluate tactical family given strategic action."""
        try:
            encoded_state = self.agent.encode_state(state)
            with torch.no_grad():
                output = self.agent.hierarchical_forward(
                    encoded_state,
                    HierarchyLevel.TACTICAL,
                    strategic_action=strategic_action,
                )
                value = output["value"].item()
                logits = output["policy_logits"]
                log_probs = F.log_softmax(logits, dim=-1)
                family_idx = TacticalFamilies.ALL_FAMILIES.index(tactic_family)
                log_prob = log_probs[0, family_idx].item()
                return value, log_prob
        except (ValueError, IndexError):
            return 0.0, -10.0
        except Exception:
            return 0.5, 0.0

    def _construct_action_from_node(
        self, node: SearchNode
    ) -> Optional[HierarchicalAction]:
        """
        Construct a HierarchicalAction object from a search node by combining
        the tactic with its generated parameters.
        """
        if node.tactic_family is None or node.strategic_action is None:
            return None

        # Dynamically select a tactic from the family based on the goal
        tactic = self._select_tactic_from_family(node.tactic_family, node.state)

        # Generate parameters for the chosen tactic
        parameters = self.agent.generate_tactic_parameters(
            node.state, node.tactic_family
        )

        return HierarchicalAction(
            strategic_action=node.strategic_action,
            tactic_family=node.tactic_family,
            specific_tactic=tactic,
            parameters=parameters,
            confidence=node.value,
        )

    def _select_tactic_from_family(self, tactic_family: str, state: TacticState) -> str:
        """
        Select a specific tactic from a family based on heuristics applied
        to the current proof state.
        """
        family_to_tactics = TacticalFamilies.get_family_metadata()
        available_tactics = family_to_tactics.get(
            tactic_family, {"tactics": ["sorry"]}
        )["tactics"]
        goal_str = str(state.pp) if hasattr(state, "pp") else ""

        # Heuristics for tactic selection
        if tactic_family == "apply_family":
            return "exact" if "=" in goal_str and len(goal_str) < 50 else "apply"
        elif tactic_family == "rewrite_family":
            return (
                "simp" if any(op in goal_str for op in ["+", "*", "-", "/"]) else "rw"
            )
        elif tactic_family == "intro_family":
            return "rintro" if "∃" in goal_str or "∧" in goal_str else "intro"
        elif tactic_family == "case_family":
            if "induction" in goal_str.lower():
                return "induction"
            elif "∨" in goal_str:
                return "split"
            else:
                return "cases"
        elif tactic_family == "automation_family":
            if any(op in goal_str for op in ["+", "*", "-", "^"]):
                return "ring"
            elif any(op in goal_str for op in ["≤", "≥", "<", ">"]):
                return "linarith"
            else:
                return "aesop"
        elif tactic_family == "structural_family":
            if "∃" in goal_str:
                return "constructor"
            elif "∨" in goal_str:
                return "left"
            else:
                return "constructor"
        elif tactic_family == "quantifier_family":
            return "exists" if "∃" in goal_str else "use!"
        else:
            # Default to the first tactic in the family list
            return available_tactics[0] if available_tactics else "sorry"

    def _prune_beam(self, beam_width: int) -> None:
        """Prune search beam to maintain size limit."""
        nodes = []
        while not self.open_nodes.empty():
            nodes.append(self.open_nodes.get())

        nodes = sorted(nodes, key=lambda x: x.value, reverse=True)[:beam_width]

        self.open_nodes = PriorityQueue()
        for node in nodes:
            self.open_nodes.put(node)
