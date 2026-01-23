# Notes

================================================================================
HYPERPARAMETER SEARCH SUMMARY
================================================================================

Total trials: 125
Hardware profile: laptop

BEST CONFIGURATION:
  Proofs/hour: 57.5 (primary metric)
  Proofs/second: 0.0160
  Success rate: 18.00%
  Avg proof time: 11.3s
  Total time: 563.2s
  Proofs: 9/50

  Parameters:
    num_workers: 12
    batch_size: 7
    num_tactics_to_expand: 13
    num_iterations: 156
    max_time: 278.3592135001262
    max_steps: 40
    proof_timeout: 300.0
    env_timeout: 72
    max_rollout_depth: 30
    mcts_type: guided_rollout
    num_epochs: 1
    num_theorems: 50
    train_epochs: 1

TOP 5 CONFIGURATIONS:

  1. proofs/hr=57.5, rate=18.00%, workers=12, batch=7, tactics=13, iters=156
  2. proofs/hr=49.0, rate=16.00%, workers=12, batch=7, tactics=13, iters=156
  3. proofs/hr=41.3, rate=10.00%, workers=12, batch=16, tactics=12, iters=100
  4. proofs/hr=39.0, rate=16.00%, workers=10, batch=14, tactics=21, iters=262
  5. proofs/hr=38.5, rate=12.00%, workers=12, batch=7, tactics=13, iters=156
================================================================================

Optimal config saved to hyperparam_results/optimal_config.json
wandb:
wandb: 🚀 View run  at:
wandb: Find logs at: wandb/run-20260122_092222-sddnccjs/logs
