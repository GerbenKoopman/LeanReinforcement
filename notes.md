# Notes

============================================================
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:921 - COORDINATE DESCENT COMPLETE
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:922 - ============================================================
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:923 - Total trials: 125
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:924 - Best score: 57.5 proofs/hour
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:925 - Optimal configuration:
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   num_workers: 10
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   batch_size: 14
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   num_tactics_to_expand: 21
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   num_iterations: 262
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   env_timeout: 99
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   max_time: 116.65631459994952
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   proof_timeout: 260.06211240030285
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   max_steps: 41
2026-01-23 17:37:09.978 | INFO     | __main__:coordinate_descent_search:927 -   max_rollout_depth: 35
2026-01-23 17:37:09.992 | INFO     | __main__:_save_results:1046 - Results saved to hyperparam_results/coordinate_descent_final.json
2026-01-23 17:37:09.992 | INFO     | __main__:_save_config:1019 - Config saved to hyperparam_results/optimal_config.json

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
