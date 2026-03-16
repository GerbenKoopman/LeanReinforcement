"""PPO package exports."""

from lean_reinforcement.agent.ppo.actor import (
    build_lora_actor,
    get_action_log_probs,
)
from lean_reinforcement.agent.ppo.critics import (
    EuclideanCritic,
    HyperbolicCritic,
)
from lean_reinforcement.agent.ppo.losses import (
    compute_critic_loss,
    compute_gae,
    compute_ppo_actor_loss,
    returns_to_bin_targets,
)
from lean_reinforcement.agent.ppo.ppo import EuclideanPPO, HyperbolicPPO, PPOConfig

__all__ = [
    "build_lora_actor",
    "get_action_log_probs",
    "EuclideanCritic",
    "HyperbolicCritic",
    "compute_critic_loss",
    "compute_gae",
    "compute_ppo_actor_loss",
    "returns_to_bin_targets",
    "PPOConfig",
    "EuclideanPPO",
    "HyperbolicPPO",
]
