#!/bin/bash

# 1. Submit the environment setup job and capture its Job ID
# --parsable makes sbatch return ONLY the job ID (e.g., 12345)
JOB_ID_1=$(sbatch --parsable environment.job)
echo "Submitted Environment Job: $JOB_ID_1"

# 2. Submit the indexing job with a dependency on Job 1
# afterok means "Run this ONLY after the previous job finished with no errors"
JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 index_corpus.job)
echo "Submitted Index Job: $JOB_ID_2 (Dependency: $JOB_ID_1)"

# 3. Submit the training job with a dependency on Job 2
JOB_ID_3=$(sbatch --parsable --dependency=afterok:$JOB_ID_2 train.job)
echo "Submitted Train Job: $JOB_ID_3 (Dependency: $JOB_ID_2)"
