cdef class Node:
    cdef public object state
    cdef public object _pp  # Cached pretty-print string for dedup
    cdef public list parents  # List of (parent_node, action) tuples for DAG structure
    cdef public object action
    cdef public float prior_p
    cdef public list children
    cdef public int visit_count
    cdef public float max_value
    cdef public bint is_terminal
    cdef public list untried_actions
    cdef public object encoder_features
    cdef public int depth  # Depth from root

    cpdef float value(self)
    cpdef bint is_fully_expanded(self)
    cpdef void add_parent(self, Node parent, object action=*)
    cpdef Node get_parent(self)
    cpdef void release_encoder_features(self)

cdef class BaseMCTS:
    cdef public object env
    cdef public object transformer
    cdef public config
    cdef public float exploration_weight
    cdef public int max_tree_nodes
    cdef public int batch_size
    cdef public int num_tactics_to_expand
    cdef public int max_rollout_depth
    cdef public float max_time
    cdef public int node_count
    cdef public dict virtual_losses
    cdef public dict seen_states
    cdef public object theorem
    cdef public object theorem_pos
    cdef public Node root
    cdef public double _search_deadline

    cpdef bint _is_timeout(self)
    cpdef int _get_virtual_loss(self, Node node)
    cpdef void _add_virtual_loss(self, Node node, int loss=*)
    cpdef void _remove_virtual_loss(self, Node node, int loss=*)
    cpdef object _get_state_key(self, object state)
    cdef Node _create_child_node(self, Node parent_node, object action, object next_state, float prior_p)
    cpdef Node _select(self, Node node)
    cpdef float _puct_score(self, Node node)
    cpdef Node _get_best_child(self, Node node)
    cpdef Node _expand(self, Node node)
    cpdef list _expand_batch(self, list nodes)
    cpdef float _simulate(self, Node node)
    cpdef list _simulate_batch(self, list nodes)
    cpdef void _backpropagate(self, Node node, float reward)
    cpdef int _count_nodes(self, Node node, set _visited=*)
    cpdef void _rebuild_seen_states(self, Node node, set _visited=*)
    cpdef float _pruning_score(self, Node node)
    cpdef void _prune_to_budget(self)
