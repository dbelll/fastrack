#!/bin/bash
#
#
#   resGPUbestx5r <num_replicate>
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=5 --MAX_TURNS=20"
_ag1="--NUM_HIDDEN=4 --NUM_AGENTS=64 --NUM_OPPONENTS=32"
_ag2="--MIN_PIECES=2 --MAX_PIECES=5"
_learn="--NUM_SESSIONS=4000 --SEGS_PER_SESSION=2 --EPISODE_LENGTH=10"
_ops="--OP_METHOD=1 --BEGIN_USING_BEST_OPS=0 --DETERMINE_BEST_OP_FREQ=400 --NUM_REPLICATE=$1"
_compete="--RR_GAMES=640 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=100 --ICHAMP=1"
_params="--MIN_ALPHA=0.00 --MAX_ALPHA=0.40 --EPSILON=0.00 --GAMMA=0.95 --MIN_LAMBDA=0.10 --MAX_LAMBDA=0.90"
_run="--GPU"

$_location/fastrack $_setup $_ag1 $_ag2 $_learn $_ops $_compete $_params $_run
