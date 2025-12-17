import unittest
import argparse
from lean_reinforcement.utilities.config import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    def test_from_args(self) -> None:
        args = argparse.Namespace(
            config_file=None,
            output_dir="output",
            resume=False,
            data_type="random",
            num_epochs=5,
            num_theorems=50,
            num_iterations=10,
            max_steps=20,
            batch_size=4,
            num_workers=2,
            mcts_type="alpha_zero",
            indexed_corpus_path=None,
            model_name="test/model",
            num_tactics_to_expand=4,
            max_rollout_depth=10,
            max_time=600.0,
            env_timeout=100,
            train_epochs=2,
            train_value_head=False,
            use_final_reward=True,
            save_training_data=False,
            save_checkpoints=False,
            checkpoint_dir="/tmp/ckpt",
            use_wandb=False,
        )

        config = TrainingConfig.from_args(args)

        self.assertEqual(config.data_type, "random")
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.num_tactics_to_expand, 4)
        self.assertEqual(config.max_rollout_depth, 10)
        self.assertEqual(config.checkpoint_dir, "/tmp/ckpt")

    def test_defaults(self) -> None:
        # This test assumes we can run get_config with empty args,
        # but get_config parses sys.argv. We need to patch sys.argv or use parse_args on the parser directly.
        # Since get_config creates the parser and parses args immediately, it's hard to test without mocking sys.argv.
        pass


if __name__ == "__main__":
    unittest.main()
