#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/darwin/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=5 --MAX_TURNS=20"
_agent="--NUM_HIDDEN=4 --NUM_AGENTS=4"
_learn="--NUM_SESSIONS=1 --EPISODE_LENGTH=2000 --WARMUP_LENGTH=0 --BENCHMARK_GAMES=0"
_params="--ALPHA=0.10 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=1.00"
_run="--GPU --CPU"

$_location/fastrack $_setup $_agent $_learn $_params $_run
