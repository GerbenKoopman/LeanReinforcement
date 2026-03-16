"""Shared constants for PPO implementations."""

from lean_reinforcement.agent.value_head.constants import ENCODER_OUTPUT_DIM

ENCODER_HIDDEN_DIM = ENCODER_OUTPUT_DIM

LATENT_DIM = 64
NUM_BINS = 51
RHO_MAX = 0.95
XI_INIT = 0.01
PPO_CLIP_EPS = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
NUM_PPO_EPOCHS = 3
MAX_STATE_TOKENS = 2300
MAX_ACTION_TOKENS = 64
