#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=2 --MAX_TURNS=5"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=64 --NUM_OPPONENTS=16"
_learn="--NUM_SESSIONS=10000 --EPISODE_LENGTH=5 --WARMUP_LENGTH=0 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=40 --BENCHMARK_OPS=40 --BEGIN_USING_BEST_OPS=0 --ICHAMP=1 --SEGS_PER_SESSION=50 --OP_METHOD=3"
_params="--ALPHA=0.20 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=0.25"
_run="--GPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run

