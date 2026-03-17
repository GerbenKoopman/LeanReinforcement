"""
Implementations of MCTS algorithms. Guided-Rollout MCTS does greedy rollout for
simulation, AlphaZero MCTS calls a trained value network for evaluation.
"""

import heapq
import math
import torch
from typing import List, Optional, Dict, Any
from loguru import logger
import time
import json
from pathlib import Path

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    empty_gpu_cache,
    get_rss_gb,
    get_available_memory_gb,
    MCTS_MIN_AVAILABLE_GB,
    MAX_WORKER_RSS_GB,
)

# Nodes with at least this many visits are protected from pruning.
_MIN_VISITS_TO_PROTECT = 4
_MAX_STATE_KEY_CHARS = 100_000


class Node:
    """
    A node in the Monte Carlo Tree Search.
    Holds state, statistics, and child nodes.
    Supports DAG structure with multiple parents for state deduplication.

    Uses ``__slots__`` to reduce per-instance memory overhead (~100 bytes
    saved per node).  With thousands of nodes this adds up.
    """

    __slots__ = (
        "state",
        "_pp",
        "parents",
        "action",
        "prior_p",
        "children",
        "visit_count",
        "max_value",
        "is_terminal",
        "untried_actions",
        "encoder_features",
        "depth",
    )

    def __init__(
        self,
        state: TacticState | ProofFinished | LeanError | ProofGivenUp,
        parent: Optional["Node"] = None,
        action: Optional[str] = None,
    ):
        self.state: TacticState | ProofFinished | LeanError | ProofGivenUp | None = (
            state
        )
        # Cache the pretty-printed string for fast lookups / dedup.
        self._pp: Optional[str] = state.pp if isinstance(state, TacticState) else None
        # Support multiple parents for DAG structure
        # Each entry is (parent_node, action_that_led_here)
        self.parents: List[tuple["Node", Optional[str]]] = []
        if parent is not None:
            self.parents.append((parent, action))
        self.action = (
            action  # Keep for compatibility (first action that created this node)
        )
        self.prior_p = 0.0

        self.children: List["Node"] = []
        self.visit_count = 0
        self.max_value = float("-inf")

        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions: Optional[List[str]] = None

        self.encoder_features: Optional[torch.Tensor] = None

        # Depth from the root — used during pruning to prefer keeping
        # shallow nodes (closer to the root) and pruning deep dead ends.
        self.depth: int = (parent.depth + 1) if parent is not None else 0

    def add_parent(self, parent: "Node", action: Optional[str] = None) -> None:
        """Add an additional parent to this node (for DAG structure)."""
        # Avoid duplicate parent-action pairs
        if not any(p == parent and a == action for p, a in self.parents):
            self.parents.append((parent, action))

    @property
    def parent(self) -> Optional["Node"]:
        """Backward compatibility: returns first parent or None."""
        return self.parents[0][0] if self.parents else None

    def value(self) -> float:
        """Calculates the value of this node. Using max_value for max-backup."""
        if self.visit_count == 0:
            return 0.0
        # Return max_value instead of mean value
        return self.max_value

    def is_fully_expanded(self) -> bool:
        """Checks if all promising actions from this node have been expanded."""
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def release_encoder_features(self) -> None:
        """Free the (potentially large) encoder features tensor."""
        self.encoder_features = None


class BaseMCTS:
    """
    A base class for MCTS, containing the shared logic for the MCTS algorithm framework.
    Subclasses must implement the expansion and simulation strategies.
    """

    def __init__(
        self,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
        config: TrainingConfig,
        exploration_weight: float = math.sqrt(2),
        max_tree_nodes: int = 1000,
        batch_size: int = 8,
        num_tactics_to_expand: int = 8,
        max_rollout_depth: int = 30,
        max_time: float = 300.0,  # Max time per MCTS search step (seconds)
        log_search_tree: bool = False,
        **kwargs,
    ):
        self.env = env
        self.transformer = transformer
        self.exploration_weight = exploration_weight
        self.max_tree_nodes = max_tree_nodes
        self.batch_size = batch_size
        self.num_tactics_to_expand = num_tactics_to_expand
        self.max_rollout_depth = max_rollout_depth
        self.max_time = max_time
        self.node_count = 0
        self.virtual_losses: Dict[Node, int] = {}
        self.log_search_tree = log_search_tree

        # Timeout tracking for search operations
        self._search_deadline: Optional[float] = None

        # State deduplication: maps state string to Node
        self.seen_states: Dict[str, Node] = {}

        # Get theorem info from the environment
        self.theorem = env.theorem
        self.theorem_pos = env.theorem_pos

        # Initialize the root node with the initial state from the env
        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = Node(state=env.current_state)
        self.node_count = 1

        # Register root state in seen_states
        if isinstance(env.current_state, TacticState):
            self.seen_states[env.current_state.pp] = self.root

    def _get_state_key(
        self, state: TacticState | ProofFinished | LeanError | ProofGivenUp
    ) -> Optional[str]:
        """
        Get a hashable key for a state for deduplication purposes.
        Returns None for terminal states (errors, proof finished) as these
        are not deduplicated.
        """
        if isinstance(state, TacticState):
            key = str(state.pp)
            if len(key) > _MAX_STATE_KEY_CHARS:
                return None
            return key
        return None

    # ------------------------------------------------------------------
    # Bounded tree pruning — visit-count aware
    # ------------------------------------------------------------------

    def _pruning_score(self, node: Node) -> float:
        """Compute a composite score for pruning decisions.

        **Lower** score  →  pruned first.  The score is designed so that:

        1. Zero-visit (speculative) leaves are the cheapest to prune.
        2. Among zero-visit nodes, deeper ones are pruned before shallow.
        3. Nodes with ``visit_count >= _MIN_VISITS_TO_PROTECT`` get a
           large bonus and are effectively protected.
        4. Nodes on the proof path (``max_value == 1.0``) are never
           eligible in the first place (filtered in collection).

        Composite:  ``visits * 1000  +  value * 100  -  depth``
        """
        v = node.visit_count
        # Protect well-explored nodes with a large bonus
        if v >= _MIN_VISITS_TO_PROTECT:
            return v * 1000.0 + node.max_value * 100.0 - node.depth
        # Low-visit: small bonus from visits + value, penalise depth
        return v * 10.0 + max(node.max_value, 0.0) * 5.0 - node.depth

    def _prune_to_budget(self) -> None:
        """Evict the lowest-scored **leaf** nodes until the tree fits
        within ``max_tree_nodes``.

        Leaf nodes (no children, not on a proven path) are collected and
        sorted by ``_pruning_score``.  The worst are removed first.
        Newly created childless parents cascade into the heap.

        Called automatically during ``search()`` whenever ``node_count``
        exceeds the budget.
        """
        if self.node_count <= self.max_tree_nodes:
            return

        target = int(self.max_tree_nodes * 0.75)  # prune 25% below budget

        # Build a min-heap of (score, id, node, parent) for pruneable leaves.
        heap: list[tuple[float, int, Node, Node]] = []
        self._collect_pruneable_leaves(self.root, heap)

        heapq.heapify(heap)

        pruned = 0
        while heap and self.node_count > target:
            _score, _nid, leaf, parent = heapq.heappop(heap)

            # The leaf may have gained children since we enqueued it,
            # or been removed from its parent by a prior iteration.
            if leaf.children or leaf not in parent.children:
                continue

            parent.children.remove(leaf)
            # Clean up seen_states
            key = leaf._pp
            if key and key in self.seen_states and self.seen_states[key] is leaf:
                del self.seen_states[key]

            # Remove DAG edges so GC can collect the node.
            leaf.parents.clear()
            leaf.encoder_features = None
            leaf.state = None  # drop TacticState reference → frees Lean state ID
            self.node_count -= 1
            pruned += 1

            # If the parent has become a childless, non-root leaf it is
            # now itself a candidate for pruning.  Push it onto the heap.
            if (
                not parent.children
                and parent is not self.root
                and parent.max_value != 1.0
            ):
                p_score = self._pruning_score(parent)
                grandparent = parent.parent
                if grandparent is not None:
                    heapq.heappush(heap, (p_score, id(parent), parent, grandparent))

        if pruned > 0:
            # Keep pruning on the fast path: only rebuild occasionally when
            # the dict has grown far beyond the live node count.
            if len(self.seen_states) > (self.node_count * 3):
                self.seen_states = {}
                self._rebuild_seen_states(self.root)
            logger.debug(
                f"Pruned {pruned} leaf nodes " f"({self.node_count} nodes remaining)"
            )

    def _collect_pruneable_leaves(
        self,
        node: Node,
        heap: list[tuple[float, int, Node, Node]],
        _visited: Optional[set[int]] = None,
    ) -> None:
        """Recursively collect leaf nodes eligible for pruning.

        Uses a visited set to guard against DAG cycles created by
        state deduplication.
        """
        if _visited is None:
            _visited = set()
        nid = id(node)
        if nid in _visited:
            return
        _visited.add(nid)

        if not node.children:
            # Leaf node — pruneable unless it's the root or on proof path.
            if node is not self.root and node.max_value != 1.0:
                parent = node.parent
                if parent is not None:
                    score = self._pruning_score(node)
                    heap.append((score, id(node), node, parent))
            return

        for child in node.children:
            self._collect_pruneable_leaves(child, heap, _visited)

    def _get_virtual_loss(self, node: Node) -> int:
        return self.virtual_losses.get(node, 0)

    def _add_virtual_loss(self, node: Node, loss: int = 1):
        self.virtual_losses[node] = self.virtual_losses.get(node, 0) + loss

    def _remove_virtual_loss(self, node: Node, loss: int = 1):
        if node in self.virtual_losses:
            self.virtual_losses[node] -= loss
            if self.virtual_losses[node] <= 0:
                del self.virtual_losses[node]

    def _is_timeout(self) -> bool:
        """Check if the search has exceeded its time limit."""
        if self._search_deadline is None:
            return False
        return time.time() > self._search_deadline

    def search(
        self,
        num_iterations: int,
        batch_size: Optional[int] = None,
        max_time: Optional[float] = None,
        search_tree_log_dir: Optional[str] = None,
    ) -> None:
        """
        Run the MCTS search for a given number of iterations with batching.

        Args:
            num_iterations: Number of MCTS iterations to run.
            batch_size: Batch size for parallel expansion/simulation.
            max_time: Maximum time in seconds for this search. If None, uses self.max_time.
            search_tree_log_dir: Directory to save search tree logs.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_time is None:
            max_time = self.max_time

        start_time = time.time()
        # Store deadline as instance var so _expand/_simulate can check it
        self._search_deadline = start_time + max_time

        # Track RSS at search start to detect runaway growth.
        _rss_at_search_start = get_rss_gb()
        # Abort if total RSS growth from search start exceeds this (GiB).
        _RSS_MAX_GROWTH = 1.5

        with torch.no_grad():
            batch_count = 0
            for iteration in range(0, num_iterations, batch_size):
                # Early stopping if solution found
                if self.root.max_value == 1.0:
                    break

                # Check time limit (more frequent check)
                if self._is_timeout():
                    logger.debug(f"MCTS search timeout after {iteration} iterations")
                    break

                current_batch_size = min(batch_size, num_iterations - iteration)
                leaves = []
                selected_leaf_ids: set[int] = set()

                # 1. Selection Phase (Batch)
                for _ in range(current_batch_size):
                    if self._is_timeout():
                        break
                    leaf = self._select(self.root)

                    if leaf.is_terminal:
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(leaf, 1.0)
                            return
                        elif isinstance(leaf.state, (LeanError, ProofGivenUp)):
                            self._backpropagate(leaf, -1.0)
                        continue

                    if not isinstance(leaf.state, TacticState):
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(leaf, 1.0)
                        else:
                            self._backpropagate(leaf, -1.0)
                        continue

                    # Skip duplicate selections within the same batch
                    leaf_id = id(leaf)
                    if leaf_id in selected_leaf_ids:
                        continue
                    selected_leaf_ids.add(leaf_id)

                    # Apply virtual loss to encourage diversity in the batch
                    self._add_virtual_loss(leaf)
                    leaves.append(leaf)

                if not leaves or self._is_timeout():
                    # Clean up virtual losses on early exit
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    if self._is_timeout():
                        break
                    continue

                # 2. Expansion Phase (with timeout awareness)
                expanded_nodes = self._expand_batch(leaves)

                # Check timeout after expansion (it can be slow)
                if self._is_timeout():
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    break

                # 3. Simulation Phase
                rewards = self._simulate_batch(expanded_nodes)

                # 4. Backpropagation Phase
                for i, leaf in enumerate(leaves):
                    self._remove_virtual_loss(leaf)
                    child = expanded_nodes[i]
                    reward = rewards[i]
                    self._backpropagate(child, reward)

                    # Free encoder features after backprop — they are only
                    # needed during expansion and can be large tensors.
                    child.release_encoder_features()

                    if reward == 1.0:
                        return

                # Prune excess nodes continuously (not just at the top)
                if self.node_count > self.max_tree_nodes:
                    self._prune_to_budget()

                # --- Memory cleanup (every 4th batch) ---
                if batch_count % 4 == 0:
                    aggressive_cleanup()
                    empty_gpu_cache()

                # --- RSS growth check ---
                rss_now = get_rss_gb()
                rss_growth = rss_now - _rss_at_search_start
                if rss_growth > _RSS_MAX_GROWTH:
                    logger.warning(
                        f"RSS grew +{rss_growth:.2f} GB since search start "
                        f"(now {rss_now:.1f} GB, started at "
                        f"{_rss_at_search_start:.1f} GB, limit "
                        f"{_RSS_MAX_GROWTH} GB). Forcing cleanup."
                    )
                    aggressive_cleanup()
                    rss_after = get_rss_gb()
                    if rss_after - _rss_at_search_start > _RSS_MAX_GROWTH:
                        logger.warning(
                            f"RSS still {rss_after:.1f} GB after cleanup "
                            f"(started at {_rss_at_search_start:.1f} GB). "
                            f"Aborting search to prevent OOM."
                        )
                        break

                # --- OOM checks (every 4th batch) ---
                if batch_count % 4 == 0:
                    avail_gb = get_available_memory_gb()
                    if avail_gb < MCTS_MIN_AVAILABLE_GB:
                        logger.warning(
                            f"System memory critically low "
                            f"({avail_gb:.1f} GB available, "
                            f"threshold {MCTS_MIN_AVAILABLE_GB} GB). "
                            f"Aborting search to prevent OOM kill."
                        )
                        break

                    rss_gb = get_rss_gb()
                    if rss_gb > MAX_WORKER_RSS_GB:
                        logger.warning(
                            f"RSS ({rss_gb:.1f} GB) exceeds hard cap "
                            f"({MAX_WORKER_RSS_GB:.1f} GB). Aborting search."
                        )
                        break

                batch_count += 1

        if self.log_search_tree and search_tree_log_dir:
            self._save_search_tree(Path(search_tree_log_dir))

    def _serialize_node(self, node: Node) -> Dict[str, Any]:
        return {
            "state": node._pp,
            "action": node.action,
            "visit_count": node.visit_count,
            "max_value": node.max_value,
            "prior_p": node.prior_p,
            "depth": node.depth,
            "children": [child._pp for child in node.children if child._pp],
        }

    def _save_search_tree(self, log_dir: Path):
        """Saves the serialized search tree to a JSON file."""
        log_dir.mkdir(parents=True, exist_ok=True)
        # Create a subdirectory for the search trees
        search_tree_dir = log_dir / "search_trees"
        search_tree_dir.mkdir(parents=True, exist_ok=True)

        def _sanitize(s: str) -> str:
            return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

        th_name = getattr(self.theorem, "full_name", "theorem")
        pos_str = str(self.theorem_pos)
        safe_name = _sanitize(th_name)[:120]
        safe_pos = _sanitize(pos_str)[:40]
        tree_file = search_tree_dir / f"search_tree_{safe_name}_{safe_pos}.json"

        nodes_to_visit = [self.root]
        visited_nodes = set()
        serialized_nodes = {}

        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if node._pp and node._pp not in visited_nodes:
                visited_nodes.add(node._pp)
                serialized_nodes[node._pp] = self._serialize_node(node)
                nodes_to_visit.extend(node.children)

        tree_data = {
            "root": self.root._pp,
            "nodes": serialized_nodes,
        }

        with open(tree_file, "w") as f:
            json.dump(tree_data, f, indent=2)

    def _select(self, node: Node) -> Node:
        """
        Phase 1: Selection
        Traverse the tree from the root, picking the best child until a leaf
        node is reached.  A visited set prevents infinite loops in DAGs
        created by state deduplication.
        """
        visited: set[int] = set()
        current = node
        while not current.is_terminal and current.is_fully_expanded():
            if not current.children:
                return current
            nid = id(current)
            if nid in visited:
                return current  # cycle detected — treat as leaf
            visited.add(nid)
            current = self._get_best_child(current)
        return current

    def _get_best_child(self, node: Node) -> Node:
        """
        Selects the best child based on the specific MCTS strategy (e.g., UCB1, PUCT).
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _expand(self, node: Node) -> Node:
        """
        Phase 2: Expansion
        This method should be implemented by subclasses. It should expand the
        tree from the given node and return the node from which to start the simulation.
        """
        raise NotImplementedError

    def _expand_batch(self, nodes: List[Node]) -> List[Node]:
        """
        Phase 2: Batch Expansion
        Default implementation calls _expand sequentially.
        Subclasses should override this for parallelism/batching.
        """
        return [self._expand(node) for node in nodes]

    def _simulate(self, node: Node, env: Optional[LeanDojoEnv] = None) -> float:
        """
        Phase 3: Simulation / Evaluation
        This method is meant to be implemented by the child classes.
        """
        raise NotImplementedError

    def _simulate_batch(self, nodes: List[Node]) -> List[float]:
        """
        Phase 3: Batch Simulation
        Default implementation calls _simulate sequentially.
        Subclasses should override this for parallelism/batching.
        """
        return [self._simulate(node) for node in nodes]

    def _backpropagate(self, node: Node, reward: float):
        """
        Phase 4: Backpropagation
        Update visit counts and value sums from the given node
        all the way back up to the root through ALL parent paths (DAG traversal).
        Uses BFS with visited set to avoid updating nodes multiple times.
        """
        from collections import deque

        visited: set[int] = set()  # Track visited nodes by id
        queue: deque[Node] = deque([node])

        while queue:
            current = queue.popleft()
            node_id = id(current)

            if node_id in visited:
                continue
            visited.add(node_id)

            # Update this node
            current.visit_count += 1
            current.max_value = max(current.max_value, reward)

            # Add all parents to the queue
            for parent, _ in current.parents:
                if id(parent) not in visited:
                    queue.append(parent)

    def get_best_action(self) -> Optional[str]:
        """
        After searching, returns the best tactic (action)
        from the root node, based on the highest visit count.
        """
        if not self.root.children:
            # If no children, we might need to generate tactics from the root
            if self.root.untried_actions is None and isinstance(
                self.root.state, TacticState
            ):
                state_str = self.root.state.pp

                self.root.untried_actions = self.transformer.generate_tactics(
                    state_str, n=self.num_tactics_to_expand
                )

            if self.root.untried_actions:
                # Fallback: if search is shallow, return a generated tactic
                return self.root.untried_actions[0]
            return None

        # If a proof was found, follow the proof path
        if self.root.max_value == 1.0:
            best_child = max(self.root.children, key=lambda c: c.max_value)
        else:
            # Select the child with the most visits (most robust)
            best_child = max(self.root.children, key=lambda c: c.visit_count)
        return best_child.action

    def extract_proof_path(self) -> Optional[List[str]]:
        """
        Extract the full sequence of tactics from root to ProofFinished.
        Returns None if no proof was found in the tree.
        """
        if self.root.max_value != 1.0:
            return None

        path: List[str] = []
        current = self.root
        while not isinstance(current.state, ProofFinished):
            # Find the child on the proof path (max_value == 1.0)
            proof_child = None
            for child in current.children:
                if child.max_value == 1.0:
                    proof_child = child
                    break
            if proof_child is None:
                return None  # Shouldn't happen if max_value == 1.0

            # Find the action leading from current to proof_child
            action = proof_child.action
            # Check parent-action pairs for DAG-deduplicated nodes
            for parent, parent_action in proof_child.parents:
                if parent is current and parent_action is not None:
                    action = parent_action
                    break

            if action is None:
                return None

            path.append(action)
            current = proof_child

        return path

    def move_root(self, action: str):
        """
        Moves the root of the tree to the child corresponding to the given action.
        This allows for subtree reuse.
        """
        found_child = None
        for child in self.root.children:
            if child.action == action:
                found_child = child
                break
            # Also check DAG parent-action pairs for deduplicated nodes
            for parent, parent_action in child.parents:
                if parent is self.root and parent_action == action:
                    found_child = child
                    break
            if found_child:
                break

        if found_child:
            old_root = self.root
            self.root = found_child
            # Clear all parent references for the new root (it becomes the root)
            self.root.parents = []

            # Break cycles in old tree to help GC
            # Remove the new root from old root's children to break cycle
            if found_child in old_root.children:
                old_root.children.remove(found_child)
            # Clear old root's parents to break upward cycles
            old_root.parents = []

            self.node_count = self._count_nodes(self.root)
            # Rebuild seen_states for the new subtree
            self.seen_states = {}
            self._rebuild_seen_states(self.root)
        else:
            # If child not found, reset the tree with the current environment state
            if not isinstance(
                self.env.current_state,
                (TacticState, ProofFinished, LeanError, ProofGivenUp),
            ):
                raise TypeError(
                    f"Invalid state type for new root: {type(self.env.current_state)}"
                )

            self.root = Node(state=self.env.current_state)
            self.node_count = 1
            # Reset seen_states with new root
            self.seen_states = {}
            if isinstance(self.env.current_state, TacticState):
                self.seen_states[self.env.current_state.pp] = self.root

    def _count_nodes(self, node: Node, _visited: Optional[set[int]] = None) -> int:
        """Recursively counts the number of nodes in the subtree.

        Uses a visited set to handle DAG dedup cycles.
        """
        if _visited is None:
            _visited = set()
        nid = id(node)
        if nid in _visited:
            return 0
        _visited.add(nid)
        count = 1
        for child in node.children:
            count += self._count_nodes(child, _visited)
        return count

    def _rebuild_seen_states(
        self, node: Node, _visited: Optional[set[int]] = None
    ) -> None:
        """Recursively rebuild seen_states dictionary from a subtree.

        Uses a visited set to handle DAG dedup cycles.
        """
        if _visited is None:
            _visited = set()
        nid = id(node)
        if nid in _visited:
            return
        _visited.add(nid)

        if node._pp is not None:
            self.seen_states[node._pp] = node
        for child in node.children:
            self._rebuild_seen_states(child, _visited)
