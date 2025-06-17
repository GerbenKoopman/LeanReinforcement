"""
Simple neural network heuristic for action value estimation.

This module provides a lightweight neural network implementation for learning
action values in theorem proving environments using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union
import numpy as np


class SimpleNeuralHeuristic(nn.Module):
    """Simple neural network for action value estimation using PyTorch."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the neural network.

        Args:
            input_size: Size of input feature vector
            hidden_size: Number of hidden layer neurons
            learning_rate: Learning rate for gradient descent
            device: Device to run the model on ('cpu', 'cuda', or torch.device)
        """
        super().__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Neural network layers
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize weights with Xavier initialization
        self._initialize_weights()

        # Move to device
        self.to(self.device)

        # Optimizer with weight decay (AdamW's main benefit)
        self.optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=0.01
        )

        # Learning rate scheduler for intelligent LR adjustment
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.8, patience=10
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # For tracking
        self.training_count = 0
        self.recent_losses = []
        self.max_loss_history = 100

    def _initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input feature vector (can be numpy array or torch tensor)

        Returns:
            Predicted value as torch tensor
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure correct device
        x = x.to(self.device)

        # Ensure proper shape (batch dimension)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.network(x)

    def train_step(self, x: Union[torch.Tensor, np.ndarray], target: float) -> float:
        """
        Train on a single example.

        Args:
            x: Input feature vector
            target: Target value

        Returns:
            Loss value
        """
        self.train()  # Set to training mode

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        prediction = self.forward(x)

        # Convert target to tensor
        target_tensor = torch.tensor([target], dtype=torch.float32, device=self.device)

        # Compute loss (ensure both tensors have same shape)
        prediction_flat = prediction.view(-1)
        loss = self.criterion(prediction_flat, target_tensor)

        # Backward pass
        loss.backward()

        self.optimizer.step()

        # Update scheduler based on average loss every 10 steps
        if self.training_count % 10 == 0 and self.training_count > 0:
            avg_loss = self.get_average_loss()
            self.scheduler.step(avg_loss)

        # Track training progress
        self.training_count += 1
        loss_value = loss.item()
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > self.max_loss_history:
            self.recent_losses.pop(0)

        return loss_value

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Make a prediction without training.

        Args:
            x: Input feature vector

        Returns:
            Predicted value
        """
        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            prediction = self.forward(x)
            return prediction.squeeze().cpu().item()

    def get_average_loss(self) -> float:
        """Get average recent loss for monitoring."""
        if not self.recent_losses:
            return 0.0
        return sum(self.recent_losses) / len(self.recent_losses)

    def save_model(self, filepath: str) -> None:
        """Save the model state dict to a file."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_count": self.training_count,
                "recent_losses": self.recent_losses,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """Load the model state dict from a file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_count = checkpoint.get("training_count", 0)
        self.recent_losses = checkpoint.get("recent_losses", [])
