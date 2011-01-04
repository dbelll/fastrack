#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=5 --MAX_TURNS=20"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=32 --NUM_OPPONENTS=8"
_learn="--NUM_SESSIONS=100 --EPISODE_LENGTH=4000 --WARMUP_LENGTH=0 --BENCHMARK_GAMES=1000 --BENCHMARK_FREQ=4"
_params="--ALPHA=0.02 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=1.00"
_run="--GPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run
