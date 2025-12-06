# LeanReinforcement

Repository for LeanReinforcement (LR), a Monte Carlo Tree Search (MCTS) agent building on the [ReProver](https://github.com/lean-dojo/ReProver) system.

## Gym-like

This repository builds on the gym-like [LeanDojo](<https://github.com/lean-dojo/LeanDojo>) interface for interacting with Lean. See the [LeanDojo documentation](https://leandojo.readthedocs.io/en/latest/index.html/) for more information on how to use it.

## Monte Carlo Tree Search

Relevant sources will be added here as the project progresses.

## TODO

- [x] Import ReProver weights to Snellius
- [x] Set up premise retrieval code
- [x] Create gym environment with LeanDojo and OpenAI Gym
- [x] Decide PPO vs MCTS => MCTS on top of ReProver
- [x] Update Snellius environment
- [x] Import LeanDojo benchmark to Snellius
- [x] Look into LeanDojo step time, value functions, etc.
- [x] Implement runner script to run MCTS agent on LeanDojo environment
- [x] Implement appropriate DataLoader training and evaluation
- [x] Implement training loop
- [x] Run experiments on Snellius
- [ ] Finish in-comment ToDo's
  - [x] Checkpoint saving and loading
  - [x] Subtree reusage
  - [ ] Reimplement tactic generator fine-tuning
- [x] Implement ValueHead Dataset creation
- [x] Set up wandb/tensorboard equivalent logging
- [ ] Write README.md
- [ ] Write paper
- [ ] Fix parallelization memory issues

## License

LeanReinforcement: An MCTS RL agent for Automatic Theorem Proving in Lean.
Copyright (C) 2025  Gerben Koopman

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
