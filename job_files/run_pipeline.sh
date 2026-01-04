#!/bin/bash

# 1. Submit the environment setup job and capture its Job ID
# --parsable makes sbatch return ONLY the job ID (e.g., 12345)
JOB_ID_1=$(sbatch --parsable environment.job)
echo "Submitted Environment Job: $JOB_ID_1"

# 2. Submit the training job with a dependency on Job 1
JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 guided_rollout_training.job)
echo "Submitted Train Job: $JOB_ID_2 (Dependency: $JOB_ID_1)"
