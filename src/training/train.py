import argparse
from loguru import logger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ReProver.common import Pos

from src.utilities.dataloader import LeanDataLoader as LeanDataLoader
from src.utilities.gym import LeanDojoEnv
from src.agent.runner import AgentRunner
from src.agent.mcts import MCTS_AlphaZero, MCTS_GuidedRollout
from src.agent.premise_selection import PremiseSelector
from src.agent.tactic_generation import TacticGenerator
from src.agent.value_head import ValueHead

# --- Custom Datasets for Training ---


class ValueHeadDataset(Dataset):
    """Dataset for (state, premises) -> value_target."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PolicyHeadDataset(Dataset):
    """Dataset for (state, premises) -> tactic_target."""

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
    if not data_buffer:
        logger.warning("Value Head training skipped: No data provided.")
        return

    logger.info(f"Training Value Head on {len(data_buffer)} samples...")
    value_head.train()  # Set model to training mode

    dataset = ValueHeadDataset(data_buffer)
    # Filter out data not meant for this model
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


def train_tactic_generator(
    tactic_generator: TacticGenerator,
    data_buffer: list,
    epochs: int = 1,
    batch_size: int = 8,
):
    """
    Trains (fine-tunes) the tactic generator on collected (state, best_tactic) data.
    This treats MCTS as an "expert" and uses supervised learning.
    """
    if not data_buffer:
        logger.warning("Tactic Generator training skipped: No data provided.")
        return

    logger.info(f"Training Tactic Generator on {len(data_buffer)} samples...")
    tactic_generator.model.train()  # Set the underlying Seq2Seq model to train mode

    dataset = PolicyHeadDataset(data_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(
        tactic_generator.model.parameters(), lr=1e-5
    )  # Lower LR for fine-tuning

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = [item["state"] for item in batch]
            premises = [item["premises"] for item in batch]
            tactic_targets = [item["tactic_target"] for item in batch]

            # Format inputs
            inputs = [f"{' '.join(p)}\n{s}" for s, p in zip(states, premises)]

            tokenized_inputs = tactic_generator.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2300,
            )
            # Format labels
            tokenized_targets = tactic_generator.tokenizer(
                tactic_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            labels = tokenized_targets.input_ids

            if torch.cuda.is_available():
                tokenized_inputs = tokenized_inputs.to("cuda")
                labels = labels.to("cuda")

            optimizer.zero_grad()

            # Forward pass (model computes loss when labels are provided)
            outputs = tactic_generator.model(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(
            f"Tactic Generator Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}"
        )

    tactic_generator.model.eval()  # Set back to evaluation mode


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

            # --- DYNAMIC MCTS SELECTION ---
            if args.mcts_type == "alpha_zero":
                mcts_class = MCTS_AlphaZero
                mcts_kwargs = {"value_head": value_head}
                logger.debug("Using MCTS_AlphaZero")
            else:  # guided_rollout
                mcts_class = MCTS_GuidedRollout
                mcts_kwargs = {}  # GuidedRollout does not need a value head
                logger.debug("Using MCTS_GuidedRollout")

            runner = AgentRunner(
                env=env,
                premise_selector=premise_selector,
                tactic_generator=tactic_generator,
                mcts_class=mcts_class,
                mcts_kwargs=mcts_kwargs,
                num_iterations=args.num_iterations,
                max_steps=args.max_steps,
            )

            success, trajectory = runner.run()
            final_reward = 1.0 if success else -1.0

            # --- PROCESS TRAJECTORY FOR TRAINING DATA ---
            all_premises = dataloader.get_premises(theorem, theorem_pos)

            for state, root_node in trajectory:
                # 1. Data for Value Head
                if args.train_value_head:
                    training_data_buffer.append(
                        {
                            "state": state.pp,
                            "premises": all_premises,
                            "value_target": final_reward,  # Use final outcome as value target
                        }
                    )

                # 2. Data for Tactic Generator (Policy Head)
                if args.train_tactic_generator and root_node.children:
                    # Find the best action taken from this state based on MCTS visits
                    best_child = max(root_node.children, key=lambda c: c.visit_count)
                    if best_child.action:  # Ensure action is not None or empty
                        training_data_buffer.append(
                            {
                                "state": state.pp,
                                "premises": all_premises,
                                "tactic_target": best_child.action,  # Use best action as target
                            }
                        )

        # --- MODEL TRAINING STEP ---
        if not training_data_buffer:
            logger.warning("No data collected in this epoch. Skipping training.")
            continue

        # --- CONDITIONAL TRAINING ---
        if args.train_value_head:
            value_data = [d for d in training_data_buffer if "value_target" in d]
            train_value_head(value_head, value_data, epochs=args.train_epochs)

        if args.train_tactic_generator:
            policy_data = [d for d in training_data_buffer if "tactic_target" in d]
            train_tactic_generator(
                tactic_generator, policy_data, epochs=args.train_epochs
            )

        # TODO: Save model checkpoints
        # value_head.save_checkpoint(...)
        # tactic_generator.save_checkpoint(...)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCTS-based Training Loop for Lean Prover"
    )
    # --- Data and MCTS Args ---
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["random", "novel_premises"],
        default="random",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of self-play/training epochs.",
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
    parser.add_argument(
        "--mcts-type",
        type=str,
        choices=["guided_rollout", "alpha_zero"],
        default="alpha_zero",
        help="Which MCTS algorithm to use for self-play.",
    )

    # --- Training Args ---
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Number of training epochs to run on collected data *per* self-play epoch.",
    )
    parser.add_argument(
        "--train-value-head",
        action="store_true",
        help="Train the value head after each epoch.",
    )
    parser.add_argument(
        "--train-tactic-generator",
        action="store_true",
        help="Train the tactic generator after each epoch.",
    )

    args = parser.parse_args()
    main(args)
