#!/bin/bash
#  Keep against the better opponents
#
#   two arguments are the number of replications and the replication noise
#
_location="./bin/linux/release"
_setup="--SEED=1001 --BOARD_SIZE=5007 --NUM_PIECES=5 --MAX_TURNS=10"
_ag1="--NUM_HIDDEN=4 --NUM_AGENTS=32 --NUM_OPPONENTS=16"
_ag2="--MIN_PIECES=2 --MAX_PIECES=7"
_learn="--NUM_SESSIONS=4000 --SEGS_PER_SESSION=4 --EPISODE_LENGTH=10"
_ops="--OP_METHOD=3 --BEGIN_USING_BEST_OPS=0"
_compete="--RR_GAMES=320 --BENCHMARK_GAMES=400 --BENCHMARK_FREQ=50 --ICHAMP=1"
_params="--MIN_ALPHA=0.01 --MAX_ALPHA=0.50 --MIN_EPSILON=.0 --MAX_EPSILON=1.0 --GAMMA=0.95 --MIN_LAMBDA=0.20 --MAX_LAMBDA=.95"
_rep="--NUM_REPLICATE=16 --REPLICATE_NOISE=$1"
_run="--GPU"

$_location/fastrack $_setup $_ag1 $_ag2 $_learn $_ops $_compete $_params $_rep $_run
