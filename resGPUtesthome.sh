#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=2 --MAX_TURNS=5"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=32 --NUM_OPPONENTS=4"
_learn="--NUM_SESSIONS=4000 --EPISODE_LENGTH=30 --WARMUP_LENGTH=0 --BENCHMARK_GAMES=100 --BENCHMARK_FREQ=400 --BEGIN_USING_BEST_OPS=1000000 --DETERMINE_BEST_OP_FREQ=100"
_params="--ALPHA=0.05 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=0.25"
_run="--GPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run

