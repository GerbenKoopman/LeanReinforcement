# LeanReinforcement

Repository for LeanReinforcement (LR), a Monte Carlo Tree Search (MCTS) agent building on the [ReProver](https://github.com/lean-dojo/ReProver) system.

## Gym-like

This repository builds on the gym-like [LeanDojo](<https://github.com/lean-dojo/LeanDojo>) interface for interacting with Lean. See the [LeanDojo documentation](https://leandojo.readthedocs.io/en/latest/index.html/) for more information on how to use it.

## Monte Carlo Tree Search

Relevant sources will be added here as the project progresses.

## TODO

- [ ] Import ReProver encoder weights to Snellius
- [x] Set up premise retrieval code
- [x] Create gym environment with LeanDojo and OpenAI Gym
- [x] Decide PPO vs MCTS => MCTS on top of ReProver
- [ ] Update Snellius environment
- [ ] Import LeanDojo benchmark to Snellius
- [ ] Look into LeanDojo step time, value functions, etc.
- [ ] Implement runner script to run MCTS agent on LeanDojo environment
- [ ] Implement appropriate DataLoader training and evaluation
- [ ] Implement training loop
- [ ] Run experiments on Snellius
