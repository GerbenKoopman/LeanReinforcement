cdef class Node:
    cdef public object state
    cdef public Node parent
    cdef public object action
    cdef public float prior_p
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
    cdef public int node_count
    cdef public dict virtual_losses
    cdef public dict seen_states
    cdef public object theorem
    cdef public object theorem_pos
    cdef public Node root

    cpdef int _get_virtual_loss(self, Node node)
    cpdef void _add_virtual_loss(self, Node node, int loss=*)
    cpdef void _remove_virtual_loss(self, Node node, int loss=*)
    cpdef Node _select(self, Node node)
    cpdef Node _get_best_child(self, Node node)
    cpdef Node _expand(self, Node node)
    cpdef list _expand_batch(self, list nodes)
    cpdef float _simulate(self, Node node)
    cpdef list _simulate_batch(self, list nodes)
    cpdef void _backpropagate(self, Node node, float reward)
    cpdef int _count_nodes(self, Node node)
