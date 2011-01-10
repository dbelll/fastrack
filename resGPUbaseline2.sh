#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=2 --MAX_TURNS=5"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=64 --NUM_OPPONENTS=32"
_learn="--NUM_SESSIONS=4000 --SEGS_PER_SESSION=2 --EPISODE_LENGTH=5 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=200 --ICHAMP=1"
_params="--ALPHA=0.20 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=0.75"
_run="--GPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run