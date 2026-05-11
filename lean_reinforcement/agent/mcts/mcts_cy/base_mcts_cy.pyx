from libc.math cimport sqrt
import heapq
import json
import time
from pathlib import Path
import torch
from typing import List, Optional, Dict
from loguru import logger
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    empty_gpu_cache,
    get_rss_gb,
    get_available_memory_gb,
    MCTS_MIN_AVAILABLE_GB,
    MAX_WORKER_RSS_GB,
)


# Nodes with at least this many visits are protected from pruning.
cdef int _MIN_VISITS_TO_PROTECT = 4
cdef int _MAX_STATE_KEY_CHARS = 100000
cdef int _MIN_MAX_UNIQUE_STATES = 5000

cdef class Node:

    def __init__(self, state, Node parent=None, action=None):
        self.state = state
        # Cache pretty-print string for fast dedup lookups
        self._pp = state.pp if isinstance(state, TacticState) else None
        # Support multiple parents for DAG structure
        # Each entry is (parent_node, action_that_led_here)
        self.parents = []
        if parent is not None:
            self.parents.append((parent, action))
        self.action = action  # Keep for compatibility (first action that created this node)
        self.prior_p = 0.0
        self.children = []
        self.visit_count = 0
        self.max_value = float("-inf")
        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions = None
        self.encoder_features = None
        self.depth = (parent.depth + 1) if parent is not None else 0

    cpdef void add_parent(self, Node parent, object action=None):
        """Add an additional parent to this node (for DAG structure)."""
        # Avoid duplicate parent-action pairs
        cdef tuple pair
        for pair in self.parents:
            if pair[0] is parent and pair[1] == action:
                return
        self.parents.append((parent, action))

    cpdef Node get_parent(self):
        """Backward compatibility: returns first parent or None."""
        if self.parents:
            return <Node>self.parents[0][0]
        return None

    @property
    def parent(self):
        """Backward compatibility property."""
        return self.get_parent()

    cpdef float value(self):
        if self.visit_count == 0:
            return 0.0
        return self.max_value

    cpdef bint is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    cpdef void release_encoder_features(self):
        """Free the (potentially large) encoder features tensor."""
        self.encoder_features = None

cdef class BaseMCTS:

    def __init__(
        self,
        env,
        transformer,
        config,
        float exploration_weight=1.41421356,
        int max_tree_nodes=1000,
        int batch_size=8,
        int num_tactics_to_expand=8,
        int max_rollout_depth=30,
        float max_time=300.0,
        bint log_search_tree=False,
        float q_weight=1.0,
        **kwargs,
    ):
        self.env = env
        self.transformer = transformer
        self.config = config
        self.exploration_weight = exploration_weight
        self.max_tree_nodes = max_tree_nodes
        self.batch_size = batch_size
        self.num_tactics_to_expand = num_tactics_to_expand
        self.max_rollout_depth = max_rollout_depth
        self.max_time = max_time
        self.node_count = 0
        self.virtual_losses = {}
        self.log_search_tree = log_search_tree
        self.q_weight = q_weight
        # Seen states dictionary for deduplication (maps state string to Node)
        self.seen_states = {}

        self.theorem = env.theorem
        self._search_deadline = 0.0  # Will be set in search()
        self.theorem_pos = env.theorem_pos

        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = Node(state=env.current_state)
        self.node_count = 1

        # Register root state in seen_states
        if isinstance(env.current_state, TacticState):
            self.seen_states[env.current_state.pp] = self.root

    cpdef int _get_virtual_loss(self, Node node):
        return self.virtual_losses.get(node, 0)

    cpdef void _add_virtual_loss(self, Node node, int loss=1):
        self.virtual_losses[node] = self.virtual_losses.get(node, 0) + loss

    cpdef void _remove_virtual_loss(self, Node node, int loss=1):
        if node in self.virtual_losses:
            self.virtual_losses[node] -= loss
            if self.virtual_losses[node] <= 0:
                del self.virtual_losses[node]

    cpdef bint _is_timeout(self):
        """Check if the search has exceeded its deadline."""
        if self._search_deadline <= 0:
            return False
        return time.time() > self._search_deadline

    cpdef object _get_state_key(self, object state):
        """Get a hashable key for a state for deduplication purposes."""
        if isinstance(state, TacticState):
            if len(state.pp) > _MAX_STATE_KEY_CHARS:
                return None
            return state.pp
        return None

    cdef Node _create_child_node(self, Node parent_node, object action, object next_state, float prior_p):
        """Create a new child node, reusing an existing node if the state is already in seen_states (DAG)."""
        cdef Node child
        cdef Node existing_node
        cdef object state_key = self._get_state_key(next_state)
        cdef int max_unique_states = max(self.max_tree_nodes * 2, _MIN_MAX_UNIQUE_STATES)

        if (
            state_key is not None
            and state_key not in self.seen_states
            and len(self.seen_states) >= max_unique_states
        ):
            return parent_node

        if state_key is not None and state_key in self.seen_states:
            existing_node = self.seen_states[state_key]
            existing_node.add_parent(parent_node, action)
            if existing_node not in parent_node.children:
                parent_node.children.append(existing_node)
            return existing_node

        child = Node(next_state, parent=parent_node, action=action)
        child.prior_p = prior_p
        parent_node.children.append(child)
        self.node_count += 1
        
        if state_key is not None:
            self.seen_states[state_key] = child
        
        return child

    def search(
        self,
        int num_iterations,
        batch_size=None,
        max_time=None,
        search_tree_log_dir=None,
    ):
        cdef int iteration
        cdef int current_batch_size
        cdef list leaves
        cdef Node leaf
        cdef list expanded_nodes
        cdef list rewards
        cdef int i
        cdef float reward
        cdef Node child
        cdef double start_time
        cdef double effective_max_time

        if batch_size is None:
            batch_size = self.batch_size
        if max_time is None:
            effective_max_time = self.max_time
        else:
            effective_max_time = max_time

        cdef int b_size = batch_size
        start_time = time.time()
        self._search_deadline = start_time + effective_max_time

        # Track RSS at search start to detect runaway growth.
        cdef float _rss_at_search_start = get_rss_gb()
        cdef float _RSS_MAX_GROWTH = 1.5

        with torch.no_grad():
            batch_count = 0
            for iteration in range(0, num_iterations, b_size):
                # Early stopping if solution found
                if self.root.max_value == 1.0:
                    break

                # Check time limit
                if self._is_timeout():
                    break

                current_batch_size = min(b_size, num_iterations - iteration)
                leaves = []
                selected_leaf_ids = set()

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

                    self._add_virtual_loss(leaf)
                    leaves.append(leaf)

                if not leaves or self._is_timeout():
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    if self._is_timeout():
                        break
                    continue

                expanded_nodes = self._expand_batch(leaves)

                if self._is_timeout():
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    break

                rewards = self._simulate_batch(expanded_nodes)

                for i in range(len(leaves)):
                    leaf = leaves[i]
                    self._remove_virtual_loss(leaf)
                    child = expanded_nodes[i]
                    reward = rewards[i]
                    self._backpropagate(child, reward)

                    # Free encoder features after backprop
                    child.release_encoder_features()

                    if reward == 1.0:
                        return

                # Continuously prune excess nodes
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
                    if get_rss_gb() - _rss_at_search_start > _RSS_MAX_GROWTH:
                        logger.warning(
                            f"RSS still {get_rss_gb():.1f} GB after cleanup "
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

    def _serialize_node(self, Node node):
        return {
            "state": node._pp,
            "action": node.action,
            "visit_count": node.visit_count,
            "max_value": node.max_value,
            "prior_p": node.prior_p,
            "depth": node.depth,
            "children": [child._pp for child in node.children if child._pp],
        }

    def _save_search_tree(self, log_dir):
        """Saves the serialized search tree to a JSON file."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
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

    cpdef Node _select(self, Node node):
        cdef Node current = node
        cdef set visited = set()
        cdef object nid
        while not current.is_terminal and current.is_fully_expanded():
            if not current.children:
                return current
            nid = id(current)
            if nid in visited:
                return current  # cycle detected — treat as leaf
            visited.add(nid)
            current = self._get_best_child(current)
        return current

    cpdef float _puct_score(self, Node node):
        cdef float q_value
        cdef float exploration
        cdef int v_loss
        cdef int visit_count
        cdef Node parent = node.get_parent()

        if parent is None:
            return 0.0

        v_loss = self._get_virtual_loss(node)
        visit_count = node.visit_count + v_loss

        if visit_count == 0:
            q_value = 0.0
        else:
            q_value = self.q_weight * (
                node.max_value - (v_loss / <float>visit_count)
            )

        exploration = (
            self.exploration_weight
            * node.prior_p
            * (sqrt(parent.visit_count) / (1 + visit_count))
        )

        return q_value + exploration

    cpdef Node _get_best_child(self, Node node):
        cdef Node child
        cdef Node best_child = None
        cdef float max_score = -1e9
        cdef float score

        if not node.children:
            raise ValueError("Node has no children")

        for child in node.children:
            score = self._puct_score(child)
            if best_child is None or score > max_score:
                max_score = score
                best_child = child
        
        return best_child

    cpdef Node _expand(self, Node node):
        raise NotImplementedError

    cpdef list _expand_batch(self, list nodes):
        return [self._expand(node) for node in nodes]

    cpdef float _simulate(self, Node node, object env=None):
        raise NotImplementedError

    cpdef list _simulate_batch(self, list nodes):
        return [self._simulate(node) for node in nodes]

    cpdef void _backpropagate(self, Node node, float reward):
        """
        Phase 4: Backpropagation
        Update visit counts and value sums from the given node
        all the way back up to the root through ALL parent paths (DAG traversal).
        Uses BFS with visited set to avoid updating nodes multiple times.
        """
        from collections import deque
        
        cdef set visited = set()
        cdef object queue = deque([node])
        cdef Node current
        cdef object node_id
        cdef tuple parent_tuple
        cdef Node parent_node
        
        while queue:
            current = queue.popleft()
            node_id = id(current)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # Update this node
            current.visit_count += 1
            if reward > current.max_value:
                current.max_value = reward
            
            # Add all parents to the queue
            for parent_tuple in current.parents:
                parent_node = <Node>parent_tuple[0]
                if id(parent_node) not in visited:
                    queue.append(parent_node)

    def get_best_action(self):
        cdef Node best_child
        if not self.root.children:
            if self.root.untried_actions is None and isinstance(
                self.root.state, TacticState
            ):
                state_str = self.root.state.pp
                self.root.untried_actions = self.transformer.generate_tactics(
                    state_str, n=self.num_tactics_to_expand
                )

            if self.root.untried_actions:
                return self.root.untried_actions[0]
            return None

        # If a proof was found, follow the proof path
        if self.root.max_value == 1.0:
            best_child = max(self.root.children, key=lambda c: c.max_value)
        else:
            # Select the child with the most visits (most robust)
            best_child = max(self.root.children, key=lambda c: c.visit_count)
        return best_child.action

    def extract_proof_path(self):
        """
        Extract the full sequence of tactics from root to ProofFinished.
        Returns None if no proof was found in the tree.
        """
        if self.root.max_value != 1.0:
            return None

        cdef list path = []
        cdef Node current = self.root
        cdef Node proof_child
        cdef object action
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
            for parent_tuple in proof_child.parents:
                if parent_tuple[0] is current and parent_tuple[1] is not None:
                    action = parent_tuple[1]
                    break

            if action is None:
                return None

            path.append(action)
            current = proof_child

        return path

    def move_root(self, str action):
        cdef Node found_child = None
        cdef Node child
        cdef Node old_root
        
        for child in self.root.children:
            if child.action == action:
                found_child = child
                break
            # Also check DAG parent-action pairs for deduplicated nodes
            for parent_tuple in child.parents:
                if parent_tuple[0] is self.root and parent_tuple[1] == action:
                    found_child = child
                    break
            if found_child is not None:
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

    cpdef int _count_nodes(self, Node node, set _visited=None):
        cdef int count
        cdef Node child
        cdef object nid
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

    cpdef void _rebuild_seen_states(self, Node node, set _visited=None):
        """Recursively rebuild seen_states dictionary from a subtree.

        Uses a visited set to handle DAG dedup cycles.
        """
        cdef Node child
        cdef object nid
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

    cpdef float _pruning_score(self, Node node):
        """Compute composite pruning score (lower = pruned first).

        Zero-visit speculative leaves score lowest.  Well-visited nodes
        (>= _MIN_VISITS_TO_PROTECT) get a large bonus and are protected.
        """
        cdef int v = node.visit_count
        cdef float mv = node.max_value

        if v >= _MIN_VISITS_TO_PROTECT:
            return v * 1000.0 + mv * 100.0 - node.depth
        return v * 10.0 + max(mv, 0.0) * 5.0 - node.depth

    cpdef void _prune_to_budget(self):
        """Evict lowest-scored leaf nodes until tree fits within budget."""
        if self.node_count <= self.max_tree_nodes:
            return

        cdef int target = int(self.max_tree_nodes * 0.75)
        cdef int pruned = 0
        cdef Node leaf, parent, grandparent
        cdef float score, p_score
        cdef object key

        # Build min-heap of pruneable leaves
        heap = []
        self._collect_pruneable_leaves_cy(self.root, heap)
        heapq.heapify(heap)

        while heap and self.node_count > target:
            _score, _nid, leaf, parent = heapq.heappop(heap)

            if leaf.children or leaf not in parent.children:
                continue

            parent.children.remove(leaf)
            key = leaf._pp
            if key and key in self.seen_states and self.seen_states[key] is leaf:
                del self.seen_states[key]

            leaf.parents.clear()
            leaf.encoder_features = None
            leaf.state = None  # free TacticState / Lean state_id
            self.node_count -= 1
            pruned += 1

            if (
                not parent.children
                and parent is not self.root
                and parent.max_value != 1.0
            ):
                grandparent = parent.get_parent()
                if grandparent is not None:
                    p_score = self._pruning_score(parent)
                    heapq.heappush(heap, (p_score, id(parent), parent, grandparent))

        if pruned > 0:
            # Keep pruning on the fast path: only rebuild occasionally when
            # the dict has grown far beyond the live node count.
            if len(self.seen_states) > (self.node_count * 3):
                self.seen_states = {}
                self._rebuild_seen_states(self.root)
            logger.debug(
                f"Pruned {pruned} leaf nodes "
                f"({self.node_count} nodes remaining)"
            )

    def _collect_pruneable_leaves_cy(self, Node node, list heap, set _visited=None):
        """Recursively collect leaf nodes eligible for pruning."""
        cdef Node child, parent
        cdef float score
        cdef object nid

        if _visited is None:
            _visited = set()
        nid = id(node)
        if nid in _visited:
            return
        _visited.add(nid)

        if not node.children:
            if node is not self.root and node.max_value != 1.0:
                parent = node.get_parent()
                if parent is not None:
                    score = self._pruning_score(node)
                    heap.append((score, id(node), node, parent))
            return

        for child in node.children:
            self._collect_pruneable_leaves_cy(child, heap, _visited)
