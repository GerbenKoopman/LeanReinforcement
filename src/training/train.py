import argparse
from loguru import logger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ReProver.common import Pos

from src.utilities.dataloader import DataLoader as LeanDataLoader
from src.utilities.gym import LeanDojoEnv
from src.agent.runner import AgentRunner
from src.agent.mcts import MCTS_AlphaZero
from src.agent.premise_selection import PremiseSelector
from src.agent.tactic_generation import TacticGenerator
from src.agent.value_head import ValueHead

# --- Custom Dataset for Training ---


class ValueHeadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Training Functions ---


def train_value_head(
    value_head: ValueHead, data_buffer: list, epochs: int = 1, batch_size: int = 32
):
    """
    Trains the value head on collected data.
    """
    logger.info(f"Training Value Head on {len(data_buffer)} samples...")
    value_head.train()  # Set model to training mode

    dataset = ValueHeadDataset(data_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(value_head.value_head.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = [item["state"] for item in batch]
            premises = [item["premises"] for item in batch]
            value_targets = torch.tensor(
                [item["value_target"] for item in batch], dtype=torch.float32
            )

            if torch.cuda.is_available():
                value_targets = value_targets.to("cuda")

            # Get features from the frozen encoder
            features = value_head._encode(
                [f"{' '.join(p)}\n{s}" for s, p in zip(states, premises)]
            )

            # Get value prediction from the trainable head
            value_preds = value_head.value_head(features).squeeze()

            loss = loss_fn(value_preds, value_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Value Head Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}")

    value_head.eval()  # Set back to evaluation mode


def train_all_models(
    premise_selector: PremiseSelector,
    tactic_generator: TacticGenerator,
    value_head: ValueHead,
    data_buffer: list,
):
    """
    Placeholder for training all three models.
    This would involve separate training steps for each model based on the collected data.
    """
    logger.info("Training all models...")
    # 1. Train Value Head (as implemented above)
    train_value_head(value_head, [d for d in data_buffer if "value_target" in d])

    # 2. Train Tactic Generator
    # This requires a different data format (state, policy_target) and loss (e.g., Cross-Entropy)
    # See ReProver/generation/main.py for a full implementation example.
    logger.warning(
        "Tactic generator training is not fully implemented in this example."
    )

    # 3. Train Premise Selector
    # This is a retrieval task, often trained with contrastive loss.
    # See ReProver/retrieval/main.py for a full implementation example.
    logger.warning(
        "Premise selector training is not fully implemented in this example."
    )


# --- Main Loop ---


def main(args):
    # --- Models ---
    premise_selector = PremiseSelector()
    tactic_generator = TacticGenerator()
    value_head = ValueHead()

    # --- DataLoader ---
    logger.info(f"Loading data from 'leandojo_benchmark_4/{args.data_type}'")
    dataloader = LeanDataLoader(
        dataset_path="leandojo_benchmark_4", data_type=args.data_type
    )
    dataloader.trace_repo()

    # --- Self-Play and Training Loop ---
    for epoch in range(args.num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_epochs}")
        training_data_buffer = []

        for thm_data in dataloader.train_data[
            : args.num_theorems
        ]:  # Limiting for demonstration
            theorem = dataloader.extract_theorem(thm_data)
            if not theorem:
                continue

            theorem_pos = Pos(*thm_data["start"])
            env = LeanDojoEnv(theorem, theorem_pos)
            mcts_kwargs = {"value_head": value_head}

            runner = AgentRunner(
                env=env,
                premise_selector=premise_selector,
                tactic_generator=tactic_generator,
                mcts_class=MCTS_AlphaZero,  # AlphaZero is needed for value head
                mcts_kwargs=mcts_kwargs,
                num_iterations=args.num_iterations,
                max_steps=args.max_steps,
            )

            success, trajectory = runner.run()
            final_reward = 1.0 if success else -1.0

            # Process trajectory for training data
            for state, root_node in trajectory:
                # Data for Value Head
                all_premises = dataloader.get_premises(theorem, theorem_pos)
                training_data_buffer.append(
                    {
                        "state": state.pp,
                        "premises": all_premises,  # Assuming premise selector needs all available ones
                        "value_target": final_reward,
                    }
                )
                # TODO: Add data processing for policy head and premise selector

        # --- Model Training Step ---
        if not training_data_buffer:
            logger.warning("No data collected in this epoch. Skipping training.")
            continue

        if args.train_all:
            train_all_models(
                premise_selector, tactic_generator, value_head, training_data_buffer
            )
        elif args.train_value_head_only:
            train_value_head(value_head, training_data_buffer)
        else:
            logger.info("No training flags specified. Skipping model training.")

        # TODO: Save model checkpoints
        # value_head.save_checkpoint(...)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCTS-based Training Loop for Lean Prover"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["random", "novel_premises"],
        default="random",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=100,
        help="Number of theorems to attempt per epoch.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of MCTS iterations per step.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Max steps per proof."
    )

    # Training flags
    parser.add_argument(
        "--train-value-head-only",
        action="store_true",
        help="Train only the value head.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all models (value, tactic, premise).",
    )

    args = parser.parse_args()
    main(args)
