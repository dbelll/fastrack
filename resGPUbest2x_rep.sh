#!/bin/bash
# Baseline comparison of GPU and CPU results
#
#   two arguments are the number of replications and the replication noise
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=2 --MAX_TURNS=5"
_ag1="--NUM_HIDDEN=4 --NUM_AGENTS=64 --NUM_OPPONENTS=64"
_ag2="--MIN_PIECES=2 --MAX_PIECES=2"
_learn="--NUM_SESSIONS=12000 --SEGS_PER_SESSION=1 --EPISODE_LENGTH=5"
_ops="--OP_METHOD=1"
_compete="--RR_GAMES=3200 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=200 --ICHAMP=1"
_params="--MIN_ALPHA=0.10 --MAX_ALPHA=0.30 --EPSILON=0.00 --GAMMA=0.95 --MIN_LAMBDA=0.60 --MAX_LAMBDA=.90"
_rep="--NUM_REPLICATE=$1 --REPLICATE_NOISE=$2"
_run="--GPU"

$_location/fastrack $_setup $_ag1 $_ag2 $_learn $_ops $_compete $_params $_rep $_run
