cdef class Node  # Forward declaration

cdef class Edge:
    cdef public object action
    cdef public float prior
    cdef public Node child
    cdef public int visit_count

cdef class Node:
    cdef public object state
    cdef public list children
    cdef public int visit_count
    cdef public float max_value
    cdef public bint is_terminal
    cdef public list untried_actions
    cdef public object encoder_features

    cpdef float value(self)
    cpdef bint is_fully_expanded(self)

cdef class BaseMCTS:
    cdef public object env
    cdef public object transformer
    cdef public float exploration_weight
    cdef public int max_tree_nodes
    cdef public int batch_size
    cdef public int num_tactics_to_expand
    cdef public int max_rollout_depth
    cdef public float max_time
    cdef public int node_count
    cdef public dict virtual_losses
    cdef public dict nodes
    cdef public object theorem
    cdef public object theorem_pos
    cdef public Node root

    cpdef int _get_virtual_loss(self, Node node)
    cpdef void _add_virtual_loss(self, Node node, int loss=*)
    cpdef void _remove_virtual_loss(self, Node node, int loss=*)
    cpdef set _get_reachable_nodes(self, Node root)
    cpdef int _prune_unreachable_nodes(self, Node new_root)
    cpdef tuple _select(self, Node node)
    cpdef Edge _get_best_edge(self, Node node)
    cpdef tuple _expand(self, Node node)
    cpdef list _expand_batch(self, list nodes)
    cpdef float _simulate(self, Node node)
    cpdef list _simulate_batch(self, list nodes)
    cpdef void _backpropagate(self, list path, float reward)
    cpdef Node _get_or_create_node(self, object state)
    cpdef int _count_nodes(self, Node node)
