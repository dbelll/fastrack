/*
 *  dc_memory.h
 *  fastrack
 *
 *  Created by Dwight Bell on 1/1/11.
 *  Copyright 2011 dbelll. All rights reserved.
 *
 
	Constant memory allocations on device.
 */

__constant__ AGENT dc_ag;

__constant__ float dc_gamma;
__constant__ unsigned dc_num_hidden;
__constant__ unsigned dc_num_pieces;
__constant__ unsigned dc_state_size;
__constant__ unsigned dc_board_size;
__constant__ unsigned dc_half_board_size;
__constant__ unsigned dc_half_hidden;
__constant__ unsigned dc_num_wgts;
__constant__ unsigned dc_wgts_stride;
__constant__ float dc_piece_ratioX;	// ratio of num_pieces to board_size
__constant__ float dc_piece_ratioO;	// ratio of num_pieces to (board_size - num_pieces)

// repitions to cover all weights
// There are board_size number of threads and num_wgts number of weights.
// To do a bulk copy or other operation on all weights, must repeate dc_reps_for_wgts times.
__constant__ unsigned dc_reps_for_wgts;		// = 1 + (num_wgts-1)/board_size

__constant__ int *dc_moves;			// pointer to d_g_moves array on device

__constant__ unsigned dc_max_turns;
__constant__ unsigned dc_episode_length;
__constant__ unsigned dc_benchmark_games;

