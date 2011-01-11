#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   no arguments
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=5 --MAX_TURNS=10"
_ag1="--NUM_HIDDEN=4 --NUM_AGENTS=256 --NUM_OPPONENTS=128"
_ag2="--MIN_PIECES=2 --MAX_PIECES=5"
_learn="--NUM_SESSIONS=8000 --SEGS_PER_SESSION=1 --EPISODE_LENGTH=5"
_ops="--OP_METHOD=1"
_compete="--RR_GAMES=640 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=100 --ICHAMP=1"
_params="--MIN_ALPHA=0.05 --MAX_ALPHA=0.50 --EPSILON=0.00 --GAMMA=0.95 --LAMBDA=0.75"
_run="--GPU"

$_location/fastrack $_setup $_ag1 $_ag2 $_learn $_ops $_compete $_params $_run
