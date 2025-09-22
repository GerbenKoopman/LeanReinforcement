import torch
from lean_dojo import TacticState

from ..agents import BaseAgent
from ...environment import StepResult
from ..transformer.data.repository import RepoManager
from .model import SimpleTransformer


from .config import SimpleTransformerConfig


class SimpleTransformerAgent(BaseAgent):
    """
    A simple transformer-based agent for Lean.
    """

    def __init__(self, config: SimpleTransformerConfig):
        super().__init__()
        self.config = config
        self.repo_manager = RepoManager(
            config.repo_url, config.repo_commit, config.build_deps
        )

        # In a real scenario, the vocab would be built from the data
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        # Pre-populate vocab with some common tokens
        for i, token in enumerate(
            "abcdefghijklmnopqrstuvwxyz0123456789()[]{},.+-*/:=_<>",
            start=len(self.vocab),
        ):
            self.vocab[token] = i
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        self.model = SimpleTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab["<pad>"])

    def _tokenize(self, text: str) -> torch.Tensor:
        # Simple whitespace tokenization
        tokens = text.split()
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(
            0
        )

    def _decode(self, token_ids: torch.Tensor) -> str:
        token_ids_np = token_ids.cpu().numpy()
        tokens = [
            self.rev_vocab.get(int(token_id), "<unk>") for token_id in token_ids_np
        ]
        return " ".join(tokens)

    def encode_state(self, state: TacticState) -> dict[str, torch.Tensor]:
        return {"tactic_state_ts": self._tokenize(state.pp)}

    def select_action(self, state: TacticState, **kwargs) -> str:
        self.model.eval()
        if not isinstance(state, TacticState):
            return "skip"

        traced_repo = self.repo_manager.get_traced_repo()
        state_tensor = self.encode_state(state)["tactic_state_ts"]

        # For generation, we start with a <sos> token
        tgt = torch.tensor([[self.vocab["<sos>"]]], device=self.device)

        with torch.no_grad():
            for _ in range(20):  # Max tactic length
                output = self.model(state_tensor, tgt)
                next_token = output.argmax(dim=-1)[:, -1].unsqueeze(1)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == self.vocab["<eos>"]:
                    break

        action_tokens = tgt.squeeze(0)
        # Exclude <sos> and <eos>
        action_str = self._decode(action_tokens[1:-1])
        return action_str if action_str else "sorry"

    def update(self, step_result: StepResult):
        # This is a simplified update. A real implementation would use a replay buffer.
        if step_result.action_result == "error":
            return

        self.model.train()
        state_tensor = self.encode_state(step_result.before_state)["tactic_state_ts"]

        # Use the executed action as the ground truth for this simple setup
        action_tensor = self._tokenize(step_result.action)

        # Prepare target for loss calculation (teacher forcing)
        tgt_input = action_tensor[:, :-1]
        tgt_output = action_tensor[:, 1:]

        # Pad if necessary
        if tgt_input.shape[1] == 0:  # Handle single-token actions
            tgt_input = torch.cat(
                [tgt_input, torch.tensor([[self.vocab["<pad>"]]], device=self.device)],
                dim=1,
            )
            tgt_output = torch.cat(
                [tgt_output, torch.tensor([[self.vocab["<pad>"]]], device=self.device)],
                dim=1,
            )

        output = self.model(state_tensor, tgt_input)

        loss = self.criterion(
            output.reshape(-1, self.config.vocab_size), tgt_output.reshape(-1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
