# Transformer-Based Hierarchical RL for Lean Theorem Proving

This module implements a transformer-based hierarchical reinforcement learning architecture for automated theorem proving in Lean 4, as described in the HRL_Guide.md design document.

## Architecture Overview

The system implements a three-level hierarchical RL architecture:

1. **Strategic Level**: High-level proof planning and goal decomposition
2. **Tactical Level**: Tactic family selection and sequence planning  
3. **Execution Level**: Parameter generation and tactic application

## Core Components

### 🔍 Attention Mechanisms (`attention.py`)

- **MathematicalAttentionEncoder**: Specialized transformer for proof states
- **MultiHeadAttention**: Optimized for mathematical text processing
- **RoPEPositionalEncoding**: Rotary positional encoding for relative positions
- **AttentionPooling**: Attention-based pooling for variable-length sequences

**Key Features:**

- Specialized attention heads for goals, hypotheses, and context
- Mathematical notation-aware processing
- Efficient handling of long proof sequences

### 🏗️ Hierarchical Policies (`hierarchy.py`)

- **HierarchicalPolicyNetwork**: Main coordinator for all three levels
- **StrategicPolicy**: High-level proof strategy selection
- **TacticalPolicy**: Tactic family selection with strategic conditioning
- **ExecutionPolicy**: Parameter generation for specific tactics

**Strategic Actions:**

- **Basic**: Induction, Contradiction, Case Analysis, Direct Proof, Rewrite/Simplify, Apply Lemma, Unfold Definition
- **Advanced**: Strong Induction, Structural Induction, Well-Founded Induction, Contrapositive, Proof by Counterexample, Constructive Proof
- **Logical**: Existential Instantiation, Universal Generalization, Disjunction Elimination, Proof by Exhaustion
- **Mathematical**: Intermediate Value Theorem, Pigeonhole Principle, Double Counting, Probabilistic Method
- **Advanced Mathematical**: Compactness Argument, Category Theory Approach, Algebraic Manipulation, Geometric Intuition, Topological Argument, Functional Analysis

**Tactical Families:**

- Apply Family (`apply`, `exact`, `refine`)
- Rewrite Family (`rw`, `simp`, `conv`)
- Intro Family (`intro`, `intros`, `rintro`)
- Case Family (`cases`, `rcases`, `induction`)
- Calc Family (`calc`, `trans`, `symm`)
- Finish Family (`sorry`, `done`, `trivial`)

### 🎯 Pointer Networks (`pointer_network.py`)

- **TacticPointerNetwork**: Attention-based tactic selection
- **ParameterPointerNetwork**: Parameter selection and generation
- **Glimpse Attention**: Multi-step attention focusing
- **Copy Mechanism**: Copy from available terms/hypotheses

**Inspired by attention-learn-to-route for:**

- Pointer attention over available tactics
- Glimpse mechanisms for complex selections
- Copy-generate hybrid approach

### 🔧 Parameter Generation (`parameter_generator.py`)

- **TacticParameterGenerator**: Main parameter generation coordinator
- **PremiseRetriever**: Retrieval from mathematical knowledge bases
- **AutoregressiveTermGenerator**: Generation of new mathematical terms
- **HypothesisSelector**: Attention-based hypothesis selection

**Tactic-Specific Generators:**

- Apply parameters: theorem/hypothesis retrieval
- Rewrite parameters: equation sequence generation
- Use parameters: witness term generation for existentials
- Simp parameters: simplification lemma selection

### 🛠️ Utilities (`utils.py`)

- **ProofStateTokenizer**: Lean-specific tokenization
- **TacticEncoder**: Tactic parsing and family mapping
- **StateEncoder**: Tensor encoding for neural networks
- **HierarchyLevel**: Enum for hierarchy levels

**Mathematical Symbol Support:**

- Unicode mathematical operators (∀, ∃, →, ∧, ∨, ¬, etc.)
- Greek letters (α, β, γ, etc.)
- Set theory symbols (∈, ∉, ⊆, ∪, ∩, etc.)
- Lean-specific keywords and tactics

### 🤖 Main Agent (`agent.py`)

- **HierarchicalTransformerAgent**: Main agent coordinating all components
- **HierarchicalSearchTree**: Best-first search with neural heuristics
- **SearchNode**: Node representation for search tree
- **HierarchicalAction**: Structured action representation

## Usage

### Basic Usage

```python
from lean_rl.agents.transformer.agent import HierarchicalTransformerAgent

# Initialize agent
agent = HierarchicalTransformerAgent(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_search_time=60.0,
    beam_width=16
)

# Use with LeanDojo environment
action = agent.select_action(tactic_state)
```

### Integration with LeanDojo

```python
from lean_dojo import Dojo, Theorem, LeanGitRepo
from lean_rl.environment import LeanEnvironment
from lean_rl.agents.transformer.agent import HierarchicalTransformerAgent

# Setup
repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "commit_hash")
theorem = Theorem(repo, "path/to/file.lean", "theorem_name")

# Create environment and agent
env = LeanEnvironment(repo)
agent = HierarchicalTransformerAgent()

# Proof loop
state = env.reset(theorem)
while not done:
    action = agent.select_action(state)
    step_result = env.step(action)
    agent.update(step_result)
    state = step_result.state
    done = step_result.done
```

### Hierarchical Action Construction

```python
# Get complete hierarchical action
hierarchical_action = agent.construct_full_action(state)

print(f"Strategic: {hierarchical_action.strategic_action}")
print(f"Tactical: {hierarchical_action.tactic_family}")  
print(f"Specific: {hierarchical_action.specific_tactic}")
print(f"Parameters: {hierarchical_action.parameters}")
```

### Custom Configuration

```python
# Custom agent configuration
agent = HierarchicalTransformerAgent(
    vocab_size=15000,          # Larger vocabulary
    d_model=768,               # Larger model
    n_heads=12,                # More attention heads
    n_layers=12,               # Deeper network
    dropout=0.1,               # Regularization
    max_search_time=120.0,     # Longer search
    beam_width=32,             # Wider beam search
    device="cuda"              # GPU acceleration
)
```

## Training

The agent can be trained using standard RL algorithms. The hierarchical structure allows for:

1. **Curriculum Learning**: Progressive difficulty from simple to complex theorems
2. **Imitation Learning**: Bootstrap from human proof demonstrations  
3. **Continual Learning**: Prevent catastrophic forgetting across mathematical domains
4. **Multi-Task Learning**: Joint training on different theorem types

### Example Training Loop

```python
# Training configuration
optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(num_epochs):
    for theorem_batch in theorem_loader:
        # Collect episodes
        episodes = []
        for theorem in theorem_batch:
            episode = collect_episode(agent, theorem)
            episodes.append(episode)
        
        # Update agent
        loss = compute_policy_loss(episodes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step(validation_performance)
```

## Model Architecture Details

### Attention Mechanisms

- **Mathematical Text Processing**: Specialized for Lean syntax and mathematical notation
- **Goal-Hypothesis Attention**: Separate attention heads for different proof components
- **Cross-Attention**: Between proof state and available tactics/terms
- **Self-Attention**: Within proof sequences for context understanding

### Hierarchical Design

- **Shared Encoder**: Common proof state encoding across all levels
- **Level-Specific Policies**: Specialized networks for each hierarchy level
- **Conditioning**: Lower levels conditioned on higher-level decisions
- **Value Networks**: Separate value estimation for each level

### Parameter Generation

- **Retrieval-Augmented**: Combines neural generation with knowledge base retrieval
- **Copy Mechanisms**: Copy from available terms, hypotheses, definitions
- **Type-Aware**: Respects Lean's type system in parameter generation
- **Autoregressive**: Sequential generation for complex parameters

## Performance Features

### Memory Efficiency

- **Gradient Checkpointing**: Reduce memory usage for large models
- **Dynamic Batching**: Efficient handling of variable-length sequences
- **Attention Sparsity**: Sparse attention for long proof sequences

### Speed Optimizations

- **Parallel Search**: Parallel evaluation of search nodes
- **Cached Encodings**: Cache proof state encodings
- **Early Stopping**: Terminate search when good solution found
- **Beam Pruning**: Efficient beam search with neural heuristics

## Integration with Existing Systems

### LeanAgent Compatibility

The transformer agent can be used as a drop-in replacement for LeanAgent's neural components while maintaining compatibility with:

- Fisher Information Matrix for continual learning
- Retrieval-augmented generation
- Best-first search infrastructure
- Distributed training setup

### LeanCopilot Integration

The agent can provide neural guidance for LeanCopilot's:

- Tactic suggestion
- Term synthesis  
- Proof search
- Interactive proving

### ReProver Integration

Compatible with ReProver's:

- Distributed training framework
- Premise selection
- Proof search algorithms
- Evaluation metrics

## Testing and Examples

To test the agent components, you can use the existing modules directly:

```python
# Test attention mechanisms
from lean_rl.agents.transformer.attention import MathematicalAttentionEncoder

encoder = MathematicalAttentionEncoder(vocab_size=10000, d_model=512)
# encoder.forward(proof_state_tokens)

# Test hierarchical policies  
from lean_rl.agents.transformer.hierarchy import HierarchicalPolicyNetwork

policy = HierarchicalPolicyNetwork(d_model=512, vocab_size=10000)
# strategic, tactical, execution = policy.forward(proof_encoding)

# Test parameter generation
from lean_rl.agents.transformer.parameter_generator import TacticParameterGenerator

param_gen = TacticParameterGenerator(d_model=512, vocab_size=10000)
# parameters = param_gen.generate_parameters(tactic, proof_state)
```

The modules demonstrate:

- Hierarchical action selection through the policy networks
- Proof state tokenization and encoding via the attention encoder
- Parameter generation for different tactic families
- Model composition and modular design
- Integration patterns for Lean environments

## Future Extensions

### Planned Features

1. **Graph Neural Networks**: Integration with mathematical knowledge graphs
2. **Multi-Modal Learning**: Natural language theorem statements
3. **Federated Learning**: Distributed training across institutions
4. **Interactive Learning**: Human-in-the-loop feedback
5. **Automated Formalization**: Natural language to formal proof

### Research Directions

1. **Mathematical Creativity**: Novel proof technique discovery
2. **Cross-System Transfer**: Lean ↔ Coq ↔ Isabelle
3. **Collaborative Proving**: Multi-agent theorem proving
4. **Educational Applications**: Proof tutoring and explanation

## Dependencies

- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- LeanDojo (for environment interaction)

## Citation

If you use this code, please cite:

```bibtex
@misc{hierarchical-lean-rl,
  title={Hierarchical Reinforcement Learning for Lean Theorem Proving},
  author={Your Name},
  year={2025},
  note={Transformer-based hierarchical RL architecture}
}
```
