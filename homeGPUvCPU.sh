#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
#_location="./bin/darwin/release"
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=2 --MAX_TURNS=5"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=1 --NUM_OPPONENTS=1"
_learn="--NUM_SESSIONS=200 --EPISODE_LENGTH=5000 --WARMUP_LENGTH=0 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=25 --ICHAMP=1"
_params="--ALPHA=0.20 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=0.75"
_run="--GPU --CPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run
