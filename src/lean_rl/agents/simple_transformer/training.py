"""
Training script for the SimpleTransformerAgent.
"""

import logging
from tqdm import tqdm
from lean_dojo import Theorem

from lean_rl.environment import LeanEnvironment
from lean_rl.agents.simple_transformer.agent import SimpleTransformerAgent

logger = logging.getLogger(__name__)

from lean_dojo import Theorem, TracedTheorem


def train(
    agent: SimpleTransformerAgent,
    env: LeanEnvironment,
    theorems: list[TracedTheorem],
    num_epochs: int,
):
    """
    Main training loop.

    Args:
        agent: The agent to train.
        env: The Lean environment.
        theorems: A list of theorems to use for training.
        num_epochs: The number of epochs to train for.
    """
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        total_reward = 0
        for theorem_traced in tqdm(theorems, desc=f"Epoch {epoch + 1}"):
            theorem = Theorem(
                theorem_traced.repo,
                theorem_traced.file_path,
                str(theorem_traced.theorem),
            )
            state = env.reset(theorem)
            done = False
            episode_reward = 0

            while not done:
                if state is None:
                    break
                action = agent.select_action(state)
                step_result = env.step(action)
                agent.update(step_result)

                state = step_result.state
                done = step_result.done
                episode_reward += step_result.reward

            total_reward += episode_reward

        logger.info(f"Epoch {epoch + 1} finished. Total reward: {total_reward}")
