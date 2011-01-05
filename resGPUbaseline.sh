#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=3 --MAX_TURNS=10"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=64 --NUM_OPPONENTS=4"
_learn="--NUM_SESSIONS=4000 --EPISODE_LENGTH=50 --WARMUP_LENGTH=0 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=200"
_params="--ALPHA=0.05 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=1.00"
_run="--GPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run
