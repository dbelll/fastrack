//
//  fastrack.cu
//  fastrack
//
//  Created by Dwight Bell on 12/13/10.
//  Copyright dbelll 2010. All rights reserved.
//

#include <cuda.h>
#include "cutil.h"

#include "fastrack.h"
#include "cuda_utils.h"
#include "rand_utils.h"
#include "misc_utils.h"
#include "cuda_rand.cu"
#include "main.h"

// global random number seeds are used for any non-agent random numbers,
static unsigned g_seeds[4] = {0, 0, 0, 0};
static PARAMS g_p;

// allowable moves expressed as delta from initial position
// maximum distance left/right is board_width/2 - 1
// maximum distance forward/backward is board_width/2 - 1
// {0, 0} is an allowable move, meaning no piece is moved (ie 'pass')
//static int g_allowable_moves[8][2] = {{-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}};
static int g_allowable_moves[8][2] = MOVES_KNIGHT;

// g_moves will be the move array of size board_size * 8 (for up to 8 possible moves from each cell)
// The values stored is the cell number where the piece would be after the move.  The board_size rows
// in the table are for all the possible starting cell numbers.  A move of -1 is not valid
static int *g_moves = NULL;

static unsigned *g_start_state = NULL;

#include "prototypes.h"
#include "dc_memory.h"
#include "wgt_pointers.h"
#include "board.h"
#include "in_output.h"
#include "results.h"

#pragma mark -
#pragma mark Helpers

// return a random number from 0 to nn-1
__device__ __host__ unsigned rand_num(unsigned nn, unsigned *seeds, unsigned stride)
{
	unsigned r = RandUniformui(seeds, stride) % nn;
	return r;
}

// Calculate agent pointers to float values based on offset from ag->wgts
// Each block of weights or eligibility traces is wgts_stride * num_agents in size
// Each parameter value is num_agents in size
void set_agent_float_pointers(AGENT *ag)
{
	ag->e = ag->wgts + g_p.wgts_stride * g_p.num_agents;
	ag->saved_wgts = ag->e + g_p.wgts_stride * g_p.num_agents;
	ag->alpha = ag->saved_wgts + g_p.wgts_stride * g_p.num_agents;
	ag->epsilon = ag->alpha + g_p.num_agents;
	ag->lambda = ag->epsilon + g_p.num_agents;
}

//	Initialize the global seeds using specified seed value.
void set_global_seeds(unsigned seed)
{
	srand(seed);
	for (int i = 0; i < 4; i++) {
		g_seeds[i] = rand();
	}
	printf("global seeds are: %u %u %u %u\n", g_seeds[0], g_seeds[1], g_seeds[2], g_seeds[3]);
}

__device__ __host__ float sigmoid(float x)
{
	return 1.0f/(1.0f + expf(-x));
}

#if GLOBAL_WGTS_FORMAT == 1
unsigned calc_num_wgts(unsigned num_hidden, unsigned board_size)
{
	return num_hidden * (2 * board_size + 3);
}
unsigned calc_wgts_stride(unsigned num_hidden, unsigned board_size)
{
	return calc_num_wgts(num_hidden, board_size);
}
#elif GLOBAL_WGTS_FORMAT == 2
unsigned calc_num_wgts(unsigned num_hidden, unsigned board_size)
{
	return num_hidden * (2 * board_size + 2) + 1;
}
unsigned calc_wgts_stride(unsigned num_hidden, unsigned board_size)
{
	return MAX_STATE_SIZE * (num_hidden + 2);
}
//#elif GLOBAL_WGTS_FORMAT == 3
//unsigned calc_num_wgts(unsigned num_hidden, unsigned board_size)
//{
//	return num_hidden * (2 * board_size + 2) + 1;
//}
//unsigned calc_wgts_stride(unsigned num_hidden, unsigned board_size)
//{
//	unsigned nw = calc_num_wgts(num_hidden, board_size);
//	printf("number of weights is %d padded out to ", nw);
//	nw = 32 * (1 + (nw-1)/32);
//	printf("%d\n", nw);
//	return nw;
//}
#endif



#pragma mark -
#pragma mark game functions

// Calculate the reward for the given state (from X's perspective)
// and set the value for the terminal flag
float reward(unsigned *state, unsigned *terminal)
{
	float reward = 0.0f;
	*terminal = 0;
	if (is_empty(O_BOARD(state))){ reward = REWARD_WIN; *terminal = 1; }
	if (is_empty(X_BOARD(state))){ reward = REWARD_LOSS; *terminal = 1; }
	return reward;
}

// Calcualte the value for a state s using the specified weights,
// storing hidden activation in the specified location and returning the output value
float val_for_state(float *wgts, unsigned *state, float *hidden, float *out)
{
//	printf("calculating value for state...\n");
	unsigned terminal;
	float r = reward(state, &terminal);
	if (terminal){
//		printf("val_for_state called on terminal state, about to bail with reward = %f\n", r);
		return r;
	}

	out[0] = 0.0f;
	
	for (unsigned iHidden = 0; iHidden < g_p.num_hidden; iHidden++) {
		// first add in the bias
//		hidden[iHidden] = -1.0f * wgts[iHidden];
		hidden[iHidden] = -1.0f * gBH(wgts, iHidden);

		// next loop update for all the input nodes
		for (int i = 0; i < g_p.board_size * 2; i++) {
			if (state[i]) {
//				hidden[iHidden] += wgts[iHidden + g_p.num_hidden * (1 + i)];
				hidden[iHidden] += gIH(wgts, i, iHidden);
			}
		}
		
		// apply the sigmoid function
		hidden[iHidden] = sigmoid(hidden[iHidden]);

		// accumulate into the output
//		out[0] += hidden[iHidden] * wgts[iHidden + g_p.num_hidden * (1 + g_p.board_size * 2)];
		out[0] += hidden[iHidden] * gHO(wgts, iHidden);
	}
	
	// finally, add the bias to the output value and apply the sigmoid function
//	out[0] += -1.0f * wgts[g_p.num_wgts - g_p.num_hidden];
	out[0] += -1.0f * gBO(wgts);
	out[0] = sigmoid(out[0]);
	return out[0];
}


// add n pieces to un-occupied cells of a board
void random_add(unsigned *board, unsigned n, unsigned *seeds, unsigned stride)
{
	while (n > 0) {
//		unsigned i = rand() % g_p.board_size;
		unsigned i = RandUniformui(seeds, stride) % g_p.board_size;
		if (!board[i]) {
			board[i] = 1;
			--n;
		}
	}
}

// generate a board with n pieces placed at random
void random_board(unsigned *board, unsigned n, unsigned *seeds, unsigned stride)
{
	// first, empty the board
	for (int i = 0; i < g_p.board_size; i++) board[i] = 0;
	
	// now add a random, non-occupied cell
	random_add(board, n, seeds, stride);
}

unsigned count_board_piecesCPU(unsigned *board)
{
	unsigned count = 0;
	for (int i = 0; i < g_p.board_size; i++) if (board[i]) ++count;
	return count;
}

// duplicate the GPU method of getting a random board to match random number useage
void random_board2(unsigned *board, unsigned n, unsigned *seeds, unsigned stride)
{
	for (int i = 0; i < g_p.board_size; i++) {
		if (g_p.piece_ratioX > RandUniform(seeds + i, stride)) board[i] = 1;
		else board[i] = 0;
	}
	
	unsigned count = count_board_piecesCPU(board);
	unsigned r;
	while (count > g_p.num_pieces) {
		r = rand_num(g_p.board_size, seeds, stride);
		if (board[r] == 1) { board[r] = 0; --count; }
	}
	while (count < g_p.num_pieces){
		r = rand_num(g_p.board_size, seeds, stride);
		if (board[r] == 0) {board[r] = 1; ++count; }
	}
}

// add n pieces randomly to an existing board, avoiding any occupied cells in the mask
void random_board_masked(unsigned *board, unsigned *mask, unsigned n, unsigned *seeds, unsigned stride)
{
	// first, copy the mask to the board
	for (int i = 0; i < g_p.board_size; i++) board[i] = mask[i];

	// now add a random, non-occupied cell
	random_add(board, n, seeds, stride);
	
	// XOR away the mask
	for (int i = 0; i < g_p.board_size; i++) board[i] ^= mask[i];
}

void random_board_masked2(unsigned *board, unsigned *mask, unsigned n, unsigned *seeds, unsigned stride)
{
	for (int i = 0; i < g_p.board_size; i++) {
		if (g_p.piece_ratioO > RandUniform(seeds + i, stride) && !mask[i]) board[i] = 1;
		else board[i] = 0;
	}
	unsigned count = count_board_piecesCPU(board);
	unsigned r;
	while (count > g_p.num_pieces) {
		r = rand_num(g_p.board_size, seeds, stride);
		if (board[r] == 1) { board[r] = 0; --count; }
	}
	while (count < g_p.num_pieces) {
		r = rand_num(g_p.board_size, seeds, stride);
		if (board[r] == 0 && mask[r] == 0) { board[r] = 1; ++count; }
	}
}

// generate a random state with n pieces for ech player
void random_state(unsigned *state, unsigned n, unsigned *seeds, unsigned stride)
{
//	random_board(X_BOARD(state), n, seeds, stride);
//	random_board_masked(O_BOARD(state), X_BOARD(state), n, seeds, stride);
	random_board2(X_BOARD(state), n, seeds, stride);
	random_board_masked2(O_BOARD(state), X_BOARD(state), n, seeds, stride);
}



// copy the starting state to the provided location
void copy_start_state(unsigned *state)
{
	bcopy(g_start_state, state, g_p.state_size * sizeof(unsigned));
}

__host__ void switch_sides(unsigned *state)
{
	for (int i = 0; i < g_p.board_size; i++) {
		unsigned temp = X_BOARD(state)[i];
		X_BOARD(state)[i] = O_BOARD(state)[i];
		O_BOARD(state)[i] = temp;
	}
}

#pragma mark -
#pragma mark Init

// build the start state with pieces filling first row on each side
void build_start_state()
{
	g_start_state = (unsigned *)calloc(2*g_p.board_size, sizeof(unsigned));
	for (int i = 0; i < g_p.board_width; i++) {
		X_BOARD(g_start_state)[i] = 1;
		O_BOARD(g_start_state)[(g_p.board_size - 1) - i] = 1;
	}
}

/*
	Build the move array, g_moves, using the allowable moves in g_allowable_moves[8][2]
	g_moves will be of size board_size * 8
*/
void build_move_array()
{
	g_moves = (int *)malloc(g_p.board_size * MAX_MOVES * sizeof(int));
	for (int row = 0; row < g_p.board_height; row++) {
		for (int col = 0; col < g_p.board_width; col++) {
			for (int m = 0; m < 8; m++) {
				unsigned iMoves = m * g_p.board_size + index4rc(row, col);
				int toCol = col + g_allowable_moves[m][0];
				int toRow = row + g_allowable_moves[m][1];
				
				if (toCol >= 0 && toCol < g_p.board_width && toRow >= 0 && toRow < g_p.board_height) {
					g_moves[iMoves] = index4rc(toRow, toCol);
				}else {
					g_moves[iMoves] = -1;
				}
			}
		}
	}
}

float rand_init_wgt(unsigned *seeds, unsigned stride)
{
	return g_p.init_wgt_min + (g_p.init_wgt_max - g_p.init_wgt_min) * RandUniform(seeds, stride);
}

// Reset all agent weights to random initial values and reset trace to 0.0f
void randomize_agent(AGENT *ag)
{
	for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
		float *wgts = ag->wgts + iAg * g_p.wgts_stride;
		float *e = ag->e + iAg * g_p.wgts_stride;
		// I->H
		for (int iI = 0; iI < g_p.state_size; iI++)
			for (int iH = 0; iH < g_p.num_hidden; iH++){
				*pgIH(wgts, iI, iH) = rand_init_wgt(g_seeds, 1);
				*pgIH(e, iI, iH) = 0.0f;
			}
		// B->H
		for (int iH = 0; iH < g_p.num_hidden; iH++){
			*pgBH(wgts, iH) = rand_init_wgt(g_seeds, 1);
			*pgBH(e, iH) = 0.0f;
		}
		// H->O
		for (int iH = 0; iH < g_p.num_hidden; iH++){
			*pgHO(wgts, iH) = rand_init_wgt(g_seeds, 1);
			*pgHO(e, iH) = 0.0f;
		}
		// B->O
		*pgBO(wgts) = rand_init_wgt(g_seeds, 1);
		*pgBO(e) = 0.0f;
	}
}

// store the paramaters in the agent data
void set_agent_params(AGENT *ag, unsigned iAg, float alpha, float epsilon, float lambda)
{
	ag->alpha[iAg] = alpha;
	ag->epsilon[iAg] = epsilon;
	ag->lambda[iAg] = lambda;
}

void freeAgentGPU(AGENT *ag)
{
	if (ag) {
		if (ag->seeds) CUDA_SAFE_CALL(cudaFree(ag->seeds));
		if (ag->states) CUDA_SAFE_CALL(cudaFree(ag->states));
		if (ag->next_to_play) CUDA_SAFE_CALL(cudaFree(ag->next_to_play));
		if (ag->wgts) CUDA_SAFE_CALL(cudaFree(ag->wgts));
		if (ag->delta_wgts) CUDA_SAFE_CALL(cudaFree(ag->delta_wgts));
		CUDA_SAFE_CALL(cudaFree(ag));
	}
}

void freeAgentCPU(AGENT *ag)
{
	if (ag) {
		if (ag->seeds) free(ag->seeds);
		if (ag->states) free(ag->states);
		if (ag->next_to_play) free(ag->next_to_play);
		if (ag->wgts) free(ag->wgts);
		free(ag);
	}
}

AGENT *init_agentsCPU(PARAMS p)
{
	// Save parameters to learning log file
	FILE *f = fopen(LEARNING_LOG_FILE, "w");
	if (!f) {
		printf("could not open LEARNING_LOG_FILE\n");
		exit(-1);
	}
	save_parameters(f);
	fclose(f);
	
	printf("init_agentsCPU...\n");

	// save the parameters in a global variable and initialize other global values
	g_p = p;					// global parameters
	set_global_seeds(p.seed);	// global seeds
	build_move_array();			// global move array
	build_start_state();		// global copy of start state

	// allocate and initialize the agent data on CPU
	AGENT *ag = (AGENT *)malloc(sizeof(AGENT));

	ag->seeds = (unsigned *)malloc(4 * p.num_agents * p.board_size * sizeof(unsigned));
	for (int i = 0; i < 4*p.num_agents * p.board_size; i++) ag->seeds[i] = rand();

	// one large allocation for all float data
	ag->wgts = (float *)malloc(p.num_agent_floats * p.num_agents * sizeof(float));
	set_agent_float_pointers(ag);	
	randomize_agent(ag);
	
	for (int i = 0; i < p.num_agents; i++) { 
		set_agent_params(ag, i, p.alpha, p.epsilon, p.lambda); 
	}
	
	ag->states = (unsigned *)malloc(p.num_agents * p.state_size * sizeof(unsigned));
	for (int i = 0; i < p.num_agents * p.state_size; i++) {
		ag->states[i] = 0;
	}
	for (int iAg = 0; iAg < p.num_agents; iAg++) {
		set_start_state(ag->states + iAg * p.state_size, p.num_pieces, ag->seeds + iAg * p.board_size * 4, p.board_size);
	}
	
	ag->next_to_play = (unsigned *)malloc(p.num_agents * sizeof(unsigned));
	for (int iAg = 0; iAg < p.num_agents; iAg++) {
		ag->next_to_play[iAg] = ranf() < 0.5f ? 0 : 1;
	}

	return ag;
}


AGENT *init_agentsGPU(AGENT *agCPU)
{
	// copy agent data to the GPU
	AGENT *agGPU = (AGENT *)malloc(sizeof(AGENT));
	agGPU->seeds = device_copyui(agCPU->seeds, 4 * g_p.num_agents * g_p.board_size);
	agGPU->states = device_copyui(agCPU->states, g_p.num_agents * g_p.state_size);
	agGPU->next_to_play = device_copyui(agCPU->next_to_play, g_p.num_agents);
	agGPU->wgts = device_copyf(agCPU->wgts, g_p.num_agent_floats * g_p.num_agents);
	set_agent_float_pointers(agGPU);
	
//	printf("agGPU->seeds %p\n", agGPU->seeds);
//	printf("agGPU->states %p\n", agGPU->states);
//	printf("agGPU->wgts %p\n", agGPU->wgts);

	// allocate room on device for saving delta wgts
	agGPU->delta_wgts = device_allocf(g_p.num_agents * g_p.num_opponents * g_p.num_wgts);
	
	// Allocate room for storing temporary won-loss information on device.
	// Temporary won-loss area stores the results for opponents run in parallel (num_opponents),
	// and also for the benchmark compete results run in parallel (benchmark_ops), so have to
	// allocate temporary area large enough to hold the greater of num_opponents and benchmark_ops.
	unsigned max_ops = (g_p.num_opponents > g_p.benchmark_ops) ? g_p.num_opponents : g_p.benchmark_ops;
	printf("num_opponents is %d, benchmark_ops is %d so max_ops is %d\n", g_p.num_opponents, g_p.benchmark_ops, max_ops);
	CUDA_SAFE_CALL(cudaMalloc(&agGPU->wl, g_p.num_agents * max_ops * sizeof(WON_LOSS)));
	
	// Agents have their own value for num_pieces and max_turns, stored in arrays on the AGENT
	// structure called training_pieces and training_turns.
	CUDA_SAFE_CALL(cudaMalloc(&agGPU->training_pieces, g_p.num_agents * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc(&agGPU->training_turns, g_p.num_agents * sizeof(unsigned)));
	unsigned *h_training_pieces = (unsigned *)malloc(g_p.num_agents * sizeof(unsigned));
	unsigned *h_training_turns = (unsigned *)malloc(g_p.num_agents * sizeof(unsigned));
	for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
		h_training_pieces[iAg] = g_p.num_pieces;
		h_training_turns[iAg] = g_p.max_turns;
//		printf("[agent %d] training pieces is %d, training turns is %d\n", iAg, h_training_pieces[iAg], h_training_turns[iAg]);
	}
	agGPU->training_pieces = device_copyui(h_training_pieces, g_p.num_agents);
	agGPU->training_turns = device_copyui(h_training_turns, g_p.num_agents);
	free(h_training_pieces); free(h_training_turns);
	
//	device_dumpui("agGPU->training_pieces", agGPU->training_pieces, g_p.num_agents, 1);
//	device_dumpui("agGPU->training_turns", agGPU->training_turns, g_p.num_agents, 1);

#ifndef USE_TEXTURE_FOR_MOVES
	// Not using texture for moves, so have to allocate room on device and copy g_moves to device.
	int *d_g_moves = device_copyi(g_moves, g_p.board_size * MAX_MOVES);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_moves", &d_g_moves, sizeof(int *)));
#endif

	// copy parameter values to constant memory
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_ag", agGPU, sizeof(AGENT)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_gamma", &g_p.gamma, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_agents", &g_p.num_agents, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_opponents", &g_p.num_opponents, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_benchmark_ops", &g_p.benchmark_ops, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_hidden", &g_p.num_hidden, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_pieces", &g_p.num_pieces, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_state_size", &g_p.state_size, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_board_size", &g_p.board_size, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_wgts", &g_p.num_wgts, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_wgts_stride", &g_p.wgts_stride, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_half_board_size", &g_p.half_board_size, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_board_bits", &g_p.board_bits, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_half_hidden", &g_p.half_hidden, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_half_opponents", &g_p.half_opponents, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_half_benchmark_ops", &g_p.half_benchmark_ops, sizeof(unsigned)));
	

	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_max_turns", &g_p.max_turns, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_episode_length", &g_p.episode_length, sizeof(unsigned)));
	
	// calculate and store on device the number of benchmark games per parallel benchmarn opponent
	unsigned benchmark_games_per_op = g_p.benchmark_games / g_p.benchmark_ops;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_benchmark_games_per_op", &benchmark_games_per_op, sizeof(unsigned)));
	
	// reps_for_wgts is the number of repitions (iterations through a loop) that must be done
	// in a kernel with board_size threads per block to copy weights between global and local memory.
	unsigned reps_for_wgts = 1 + (g_p.num_wgts - 1) / g_p.board_size;
	printf("reps_for_wgts on GPU is %d\n", reps_for_wgts);
	cudaMemcpyToSymbol("dc_reps_for_wgts", &reps_for_wgts, sizeof(unsigned));
	
	float piece_ratioX = (float)g_p.num_pieces / (float)g_p.board_size;
	float piece_ratioO = (float)g_p.num_pieces / (float)(g_p.board_size - g_p.num_pieces);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_piece_ratioX", &piece_ratioX, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_piece_ratioO", &piece_ratioO, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_init_wgt_min", &g_p.init_wgt_min, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_init_wgt_max", &g_p.init_wgt_max, sizeof(float)));
	

	// Till the opponent grid, opgrid, with values [0, num_opponents) and copy to device
	// These values will be over-written prior to learning based on parameters
	for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
		for (int iOp = 0; iOp < g_p.num_opponents; iOp++) {
			g_p.opgrid[iSeg * g_p.num_opponents + iOp] = iOp;
		}
	}
	g_p.d_opgrid = device_copyui(g_p.opgrid, g_p.num_opponents * g_p.segs_per_session);
	
//	printf("default opgrid:\n");
//	dump_opponent_array();
	
#ifdef USE_TEXTURE_FOR_MOVES
	// initialize texture to hold the moves array
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;
	texRef.filterMode = cudaFilterModePoint;
	texRef.normalized = false;
	CUDA_SAFE_CALL(cudaMallocArray((cudaArray **)&d_moves, &channelDesc, g_p.board_size, MAX_MOVES));
	CUDA_SAFE_CALL(cudaMemcpyToArray(d_moves, 0, 0, g_moves, MAX_MOVES * g_p.board_size * sizeof(int), cudaMemcpyHostToDevice));
	cudaBindTextureToArray(texRef, d_moves, channelDesc);
#endif
	
	return agGPU;
}

#pragma mark -
#pragma mark CPU - run

// Make a random X move, updating state
float random_move(unsigned *state, unsigned *seeds, unsigned stride)
{
//	printf("random_move...\n");
	static unsigned *possible_moves = NULL;
	static unsigned allocated = 0;
	if (!possible_moves){
		allocated = 100;
		possible_moves = (unsigned *)malloc(allocated * 2 * sizeof(unsigned));
	}
	unsigned move_count = 0;
	
	// loop through all the possible piece positions
	for (int iFrom = 0; iFrom < g_p.board_size; iFrom++) {
		if (X_BOARD(state)[iFrom]) {
			// found a piece that might be able to move, loop over all possible moves
//			printf("found a piece that can move!\n");
			for (int m = 0; m < MAX_MOVES; m++) {
				int iTo = g_moves[m * g_p.board_size + iFrom];
				if (iTo >= 0 && !X_BOARD(state)[iTo]) {
					// found a possible move, save it
					if (move_count == allocated) {
						// need to grow the possible moves list
						allocated *= 2;
						possible_moves = (unsigned *)realloc(possible_moves, allocated * 2 * sizeof(unsigned));
					}
					possible_moves[move_count * 2] = iFrom;
					possible_moves[move_count * 2 + 1] = iTo;
					++move_count;
				}
			}
		}
	}
	
	unsigned r = (unsigned) (move_count * RandUniform(seeds, stride));
	unsigned iRandFrom = possible_moves[r*2];
	unsigned iRandTo = possible_moves[r*2 + 1];
	// do the random move and return the value
	X_BOARD(state)[iRandFrom] = 0;
	X_BOARD(state)[iRandTo] = 1;
	O_BOARD(state)[iRandTo] = 0;
	
	return 0.0f;
}


/*
	Choose the move for player X from the given state using the nn specified by wgts.
	The best move is the legal move with the vest value, based on the NN.  The value of the
	best move is returned by the function and state is updated for the move.
*/
float choose_move(unsigned *state, float *wgts, float *hidden, float *out)
{
//	printf("choose_move...\n");
	
	unsigned terminal;
	float r = reward(state, &terminal);
	if (terminal) {
		return r;
	}
	unsigned noVal = 1;
	float bestVal = 0.0f;
	unsigned iBestFrom = 0;
	unsigned iBestTo = 0;
	unsigned move_count = 0;
	
	// loop through all the possible piece positions
	for (int iFrom = 0; iFrom < g_p.board_size; iFrom++) {
		if (X_BOARD(state)[iFrom]) {
			// found a piece that might be able to move, loop over all possible moves
//			printf("found a piece that can move!\n");
			for (int m = 0; m < MAX_MOVES; m++) {
				int iTo = g_moves[m * g_p.board_size + iFrom];
				if (iTo >= 0 && !X_BOARD(state)[iTo]) {
					// found a possible move, modify the board and calculate the value
					++move_count;
					unsigned oPiece = O_BOARD(state)[iTo];	// remember if there was an O here
					X_BOARD(state)[iFrom] = 0;
					X_BOARD(state)[iTo] = 1;
					O_BOARD(state)[iTo] = 0;
					float val = val_for_state(wgts, state, hidden, out);
//					printf("possible move with value %9.4f:\n", val);
//					dump_state(state);
					if (noVal || val > bestVal) {
//						printf("Best so far !!!\n");
						// record the best move so far
						iBestFrom = iFrom;
						iBestTo = iTo;
						bestVal = val;
						noVal = 0;
					}
					// restore the state
					X_BOARD(state)[iFrom] = 1;
					X_BOARD(state)[iTo] = 0;
					O_BOARD(state)[iTo] = oPiece;
				}
			}
		}
	}
	// do the best move and return the value
	X_BOARD(state)[iBestFrom] = 0;
	X_BOARD(state)[iBestTo] = 1;
	O_BOARD(state)[iBestTo] = 0;
	
//	printf("best move with value %9.4f:\n", bestVal);
//	dump_state(state);
//	printf("\n\n");
	// recalculate to fill in hidden and out for the chosen move
	return val_for_state(wgts, state, hidden, out);
}

float take_action2(unsigned *state, float *owgts, float *hidden, float *out, unsigned *terminal, unsigned ohidden)
{
//	printf("on entry g_p.num_hidden = %d, ohidden = %d\n", g_p.num_hidden, ohidden);

	float r = reward(state, terminal);
	if (*terminal) return r;	// given state is terminal, just return the reward

	// set param consistent with opponent
	unsigned save_hidden = g_p.num_hidden;
	unsigned save_num_wgts = g_p.num_wgts;
	g_p.num_hidden = ohidden;
//	g_p.num_wgts = g_p.num_hidden * (2 * g_p.board_size + 3);
	g_p.num_wgts = calc_num_wgts(g_p.num_hidden, g_p.board_size);
	
	switch_sides(state);
//	printf("state after switching sides:\n");
//	dump_state(state);
	choose_move(state, owgts, hidden, out);
//	printf("state after opponent move:\n");
//	dump_state(state);
	switch_sides(state);
//	printf("state after switching sides again:\n");
//	dump_state(state);

	// restore normal parameters
	g_p.num_hidden = save_hidden;
	g_p.num_wgts = save_num_wgts;
	
//	printf("on exit g_p.num_hidden = %d, ohidden = %d\n", g_p.num_hidden, ohidden);
	return reward(state, terminal);
}

// O take an action from the specified state, returning the reward
float take_action(unsigned *state, float *owgts, float *hidden, float *out, unsigned *terminal)
{
	float r = reward(state, terminal);
	if (*terminal) return r;	// given state is terminal, just return the reward
	switch_sides(state);
//	printf("state after switching sides:\n");
//	dump_state(state);
	choose_move(state, owgts, hidden, out);
//	printf("state after opponent move:\n");
//	dump_state(state);
	switch_sides(state);
//	printf("state after switching sides again:\n");
//	dump_state(state);
	return reward(state, terminal);
}

// O takes a random action from the specified state, returning X's reward
float take_random_action(unsigned *state, unsigned *terminal, unsigned *seeds, unsigned stride)
{
	float r = reward(state, terminal);
	if (*terminal) return r;	// given state is terminal, just return the reward
	switch_sides(state);
//	printf("state after switching sides:\n");
//	dump_state(state);
	random_move(state, seeds, stride);
//	printf("state after opponent move:\n");
//	dump_state(state);
	switch_sides(state);
//	printf("state after switching sides again:\n");
//	dump_state(state);
	return reward(state, terminal);
}

void set_start_state(unsigned *state, unsigned pieces, unsigned *seeds, unsigned stride)
{
//	printf("set_start_state...\n");
	(pieces == 0) ? copy_start_state(state) : random_state(state, pieces, seeds, stride);
}

void reset_trace(float *e)
{
//	for (int i = 0; i < g_p.num_wgts; i++) {
//		e[i] = 0.0f;
//	}
	for (int iH = 0; iH < g_p.num_hidden; iH++) {
		*pgBH(e, iH) = 0.0f;							// B->H
		for (int iI = 0; iI < g_p.state_size; iI++) {
			*pgIH(e, iI, iH)  = 0.0f;					// I->H
		}
		*pgHO(e, iH) = 0.0f;							// H->O
	}
	*pgBO(e) = 0.0f;									// B->O
}

void update_wgts(float alpha, float delta, float *wgts, float *e)
{
//	for (int i = 0; i < g_p.num_wgts; i++) {
//		wgts[i] += alpha * delta * e[i];
//	}
	for (int iH = 0; iH < g_p.num_hidden; iH++) {
		*pgBH(wgts, iH) += alpha * delta * gBH(e, iH);						// B->H
		for (int iI = 0; iI < g_p.state_size; iI++) {
			*pgIH(wgts, iI, iH) += alpha * delta * (*pgIH(e, iI, iH));		// I->H
		}
		*pgHO(wgts, iH) += alpha * delta * gHO(e, iH);						// H->O
	}
//	printf("BO wgts, before: %9.6f\n", *pgBO(wgts));
	*pgBO(wgts) += alpha * delta * gBO(e);									// B->O
//	printf("BO wgts, after: %9.6f\n", *pgBO(wgts));
}

// update eligibility traces using the activation values for hidden and output nodes
void update_trace(unsigned *state, float *wgts, float *e, float *hidden, float *out, float lambda)
{
//#ifdef DUMP_MOVES
//	printf("update_trace\n");
//#endif
	// first decay all existing values
//	for (int i = 0; i < g_p.num_wgts; i++) {
//		e[i] *= g_p.gamma * lambda;
//	}
	for (int iH = 0; iH < g_p.num_hidden; iH++) {
		*pgBH(e, iH) *= g_p.gamma * lambda;				// B->H
		for (int iI = 0; iI < g_p.state_size; iI++) {
			*pgIH(e, iI, iH)  *= g_p.gamma * lambda;	// I->H
		}
		*pgHO(e, iH) *= g_p.gamma * lambda;				// H->O
	}
	*pgBO(e) *= g_p.gamma * lambda;						// B->O
	
	// next update the weights from hidden layer to output node
	// first the bias
	float g_prime_i = out[0] * (1.0f - out[0]);
//	printf("out[0] is %9.4f and g_prime(out) is %9.4f\n", out[0], g_prime_i);
//	unsigned iH2O = (2 * g_p.board_size + 1) * g_p.num_hidden;
//	e[iH2O + g_p.num_hidden] += -1.0f * g_prime_i;
	*pgBO(e) += -1.0f * g_prime_i;
	
	// next do all the hidden nodes to output node
	for (int j = 0; j < g_p.num_hidden; j++) {
//		e[iH2O + j] += hidden[j] * g_prime_i;
		*pgHO(e, j) += hidden[j] * g_prime_i;
//		printf("hidden node %d, activation is %9.4f, increment to e is %9.4f, new e is %9.4f\n", j, hidden[j], g_prime_i*hidden[j], e[iH2O + j]);
	}
	
	// now update the weights to the hidden nodes
	for (int j = 0; j < g_p.num_hidden; j++) {
//		float g_prime_j = hidden[j]*(1.0f - hidden[j]) * wgts[iH2O + j] * g_prime_i;
		float g_prime_j = hidden[j]*(1.0f - hidden[j]) * gHO(wgts, j) * g_prime_i;
		// first the bias to the hidden node
//		e[j] += -1.0f * g_prime_j;
		*pgBH(e, j) += -1.0f * g_prime_j;
		
		// then all the input -> hidden values
		for (int k = 0; k < g_p.board_size * 2; k++) {
//			if (state[k]) e[(k+1)*g_p.num_hidden + j] += g_prime_j;
			if (state[k]) *pgIH(e, k, j) += g_prime_j;
		}
	}
}


WON_LOSS compete(float *ag1_wgts, const char *name1, float *ag2_wgts, const char *name2, unsigned start_pieces, unsigned num_games, unsigned turns_per_game, unsigned show, unsigned ag2_hidden, unsigned *seeds, unsigned stride)
{
//	printf("\n=================================================================\n");
//	printf("         %s     vs.     %s\n", name1, name2);
	// play a numbr of games, ag1_wgts vs ag2_wgts
	// set up the starting state
	unsigned max_nh = g_p.num_hidden > ag2_hidden ? g_p.num_hidden : ag2_hidden;
	unsigned *state = (unsigned *)malloc(g_p.state_size * sizeof(unsigned));
	float *hidden = (float *)malloc(max_nh * sizeof(float));
	float *out = (float *)malloc(max_nh * sizeof(float));
	
	WON_LOSS wl;
	wl.games = num_games;
	wl.wins = 0;
	wl.losses = 0;
	unsigned turn = 0;
	unsigned game = 0;
	unsigned terminal;
	
	SHOW printf(  "-----------------------------------------------------------------\n");
	SHOW printf("game %d:\n", game);

	set_start_state(state, start_pieces, seeds, stride);
	if (RandUniform(seeds, stride) < 0.50f) {
		SHOW printf("New game, O plays first\n");
		SHOW dump_state(state, turn, 1);
		(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal)
					: take_random_action(state, &terminal, seeds, stride));
		++turn;
	}else {
		SHOW printf("New game, X plays first\n");
	}
	SHOW dump_state(state, turn, 0);

	
	float V = (ag1_wgts	? choose_move(state, ag1_wgts, hidden, out)
					: random_move(state, seeds, stride));
	
	while (game < num_games) {
		SHOW dump_state(state, turn, 1);
		float reward = (ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal)
									: take_random_action(state, &terminal, seeds, stride));
		++turn;
		SHOW dump_state(state, turn, 0);
		
		if (terminal){
			if (game_won(reward)) { ++wl.wins; SHOW printf("*** game won ***\n");}
			else { ++wl.losses; SHOW printf("*** game lost ***\n");}
		}
		
		if (terminal || (turn == turns_per_game)) {
			SHOW if (!terminal) printf("*** turn limit reached ***\n");
			if (++game < num_games){
				// get ready for next game
				SHOW printf(  "-----------------------------------------------------------------\n");
				SHOW printf("\ngame %d:\n", game);
				turn = 0;
				terminal = 0;
				set_start_state(state, start_pieces, seeds, stride);
				if (RandUniform(seeds, stride) < 0.50f) {
					SHOW printf("New game, O plays first\n");
					SHOW dump_state(state, turn, 1);
					(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal)
								: take_random_action(state, &terminal, seeds, stride));
					++turn;
//					SHOW dump_state(state, turn, 0);
				}else {
					SHOW printf("New game, X plays first\n");
				}
				SHOW dump_state(state, turn, 0);
			}
		}

		float V_prime = (ag1_wgts	? choose_move(state, ag1_wgts, hidden, out)
								: random_move(state, seeds, stride));
		V = V_prime;
	}
//	printf("[COMPETE],%5d, %8s,%5d, %8s,%5d", num_games, name1, wl.wins, name2, wl.losses);
//	printf("%7d, %7d, %7d\n", wins, losses, wins-losses);
	free(state);
	free(hidden);
	free(out);
	return wl;
}


// Run a learning session using agent ag1 against ag2.  ag2 may be NULL which represents a random player
// Start with a random board with start_pieces per side, or the normal starting board if start_pieces is 0
// Stop the learning after num_turns (for each player)
WON_LOSS auto_learn(AGENT *ag1, unsigned iAg, float *ag2_wgts, unsigned start_pieces, unsigned num_turns, unsigned max_turns)
{
//	printf("auto_learning: %d pieces,  %d turns, with %d max turns per game...\n", start_pieces, num_turns, max_turns);

	float *ag1_wgts = ag1->wgts + iAg * g_p.wgts_stride;	// points to start of learning agent's weights
	float *ag1_e = ag1->e + iAg * g_p.wgts_stride;			// points to start of learning agent's trace
	
	if (!ag1) {
		printf("***ERROR *** random agent can not learn!!!\n");
		exit(-1);
	}
	
	WON_LOSS wl = {0, 0, 0, 0};
	
//#ifdef DUMP_ALL_AGENT_UPDATES
//	dump_agent(ag1, iAg, 1);
//#endif
	
	static float *hidden = NULL;
	static float *out = NULL;
	if(!hidden) hidden = (float *)malloc(g_p.num_hidden * sizeof(float));
	if(!out) out = (float *)malloc(g_p.num_hidden * sizeof(float));

	unsigned *state = ag1->states + iAg * g_p.state_size;
	unsigned *seeds = ag1->seeds + iAg * g_p.board_size * 4;
	
	unsigned turn = 0;			// turn is incremented after player O moves
	unsigned total_turns = 0;
	unsigned terminal = 0;
	unsigned new_terminal = 0;
	
	// set up the starting state
	// already set in initialization on CPU
	set_start_state(state, start_pieces, seeds, g_p.board_size);

//	return wl;
	
#ifdef DUMP_MOVES
	printf("--------------- game %d ---------------------\n", wl.games);
#endif

	if (RandUniform(seeds, g_p.board_size) < 0.50f) {
//	if (ag1->next_to_play[iAg]) {

#ifdef DUMP_MOVES
		printf("New game, O to play first...\n");
		dump_state(state, turn, 1);		
#endif
		float r = (ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal) 
							: take_random_action(state, &terminal, seeds, g_p.board_size));
		++turn;
	}else {
#ifdef DUMP_MOVES
		printf("New game, X to play first...\n");
#endif
	}

#ifdef DUMP_MOVES
	dump_state(state, turn, 0);		
#endif

//	return wl;		// exit 0

	// choose the action, storing the next state in agCPU->state and returning the value for the next state
	float V = choose_move(state, ag1_wgts, hidden, out);

	update_trace(state, ag1_wgts, ag1_e, hidden, out, ag1->lambda[iAg]);

//	return wl;		// exit 1

#ifdef DUMP_ALL_AGENT_UPDATES
	printf("after updating trace...\n");
	dump_agent(ag1, iAg, 1);
#endif
	
	// loop over the number of turns

//	printf("turn is %d, total completed turns is %d\n", turn, total_turns);

	while (total_turns++ < num_turns) {

#ifdef DUMP_MOVES
		dump_state(state, turn, 1);
#endif

//		printf("\n\n------- %d turns left -------\n", num_turns+1);
//		printf("after own move, turn %d:\n", turn);
//		dump_state(state);		

#ifdef DUMP_ALL_AGENT_UPDATES
		printf("hidden activation values are:\n");
		for (int i = 0; i < g_p.num_hidden; i++) {
			printf(i == 0 ? "%9.4f" : ", %9.4f", hidden[i]);
		}
		printf("\n");
		printf("output value is %9.4f\n", out[0]);
#endif

		

		float reward = (ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal) 
									: take_random_action(state, &terminal, seeds, g_p.board_size));
		++turn;
		
//		break;	// exit 2;
		
#ifdef DUMP_MOVES
		dump_state(state, turn, 0);		
#endif
		
//		printf("after opponent move, turn %d:\n", turn++);
//		dump_state(state);
		
		if (terminal){
			if (game_won(reward)) ++wl.wins;
			else ++wl.losses;
#ifdef DUMP_MOVES
			printf("\n\n****** GAME OVER after %d turns with r = %9.4f *******\n", turn, reward);
			printf("record is now %d - %d\n\n\n", wl.wins, wl.losses);
#endif
		}
		
//		if (turn == max_turns) {
//			terminal = 1;
//			reward = REWARD_TIME_LIMIT;
//		}
		if (terminal || (turn >= max_turns)) {

//			break;	// exit 5

#ifdef DUMP_MOVES
			if (!terminal) printf("****** GAME OVER: reached maximum number of turns per game (turn = %d, total_turns = %d, games = %d)\n", turn, total_turns, wl.games);
#endif
			++wl.games;
//			if (++wl.games < num_turns) {
#ifdef DUMP_MOVES
				printf("\n--------------- game %d ---------------------\n", wl.games);
#endif
				turn = 0;
				float r = RandUniform(seeds, g_p.board_size);	// ordering of random number generation
																// is consistent with GPU
				set_start_state(state, start_pieces, seeds, g_p.board_size);
				
//				break;	// exit 6
				
				if (r < 0.50f) {
#ifdef DUMP_MOVES
					printf("New game, O to play first...\n");
					dump_state(state, turn, 1);		
#endif
					(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &new_terminal) 
								: take_random_action(state, &terminal, seeds, g_p.num_agents));
					++turn;
				}else {
#ifdef DUMP_MOVES
					printf("New game, X to play first...\n");
#endif
				}

#ifdef DUMP_MOVES
				dump_state(state, turn, 0);		
#endif
//			}

//			break;	// exit 7

		}
//		printf("choosing next move...\n");
		float V_prime = choose_move(state, ag1_wgts, hidden, out);

//		if (wl.games > 0) break;	// exit 8

		float delta = reward + (terminal ? 0.0f : (g_p.gamma * V_prime)) - V;

//		ag1->epsilon[iAg] = reward;	// stash delta in epsilon for debugging
//		break;	// exit 3

//		if (wl.games > 0){
//			ag1->epsilon[iAg] = delta;
//			break;					// exit 9
//		}

//		printf("updating wgts...\n");
		update_wgts(ag1->alpha[iAg], delta, ag1_wgts, ag1_e);

#ifdef DUMP_ALL_AGENT_UPDATES
		printf("reward = %9.4f, V_prime = %9.4f, V = %9.4f, delta = %9.4f\n", reward, V_prime, V, delta);
		printf("after updating weights:\n");
		dump_agent(ag1, iAg, 1);
#endif

		if (terminal) reset_trace(ag1_e);
//		printf("updating trace...\n");
		update_trace(state, ag1_wgts, ag1_e, hidden, out, ag1->lambda[iAg]);

//		ag1->epsilon[iAg] = reward;	// stash delta in epsilon for debugging
//		break;	// exit 4

//		if (wl.games > 0){
//			ag1->epsilon[iAg] = delta;
//			break;					// exit 10
//		}

#ifdef DUMP_ALL_AGENT_UPDATES
		printf("after updating trace:\n");
		dump_agent(ag1, iAg, 1);
#endif
		
		V = V_prime;
//		printf("turn is %d, total_turns is %d\n", turn, total_turns);
	}
//	printf("learning over...  ");
//	printf("W:%7d  L:%7d  D:%7d\n", wl.wins, wl.losses, wl.games - wl.wins - wl.losses);

//	free(state);
//	free(hidden);
//	free(out);
	return wl;
}


// Do a learning run on the CPU using agents in agCPU and benchmark weights in champ_wgts
RESULTS *runCPU(AGENT *agCPU, float *champ_wgts)
{
	printf("\n\n--------------------------------------------------------------\nrunning on CPU...\n");
	
#ifdef DUMP_INITIAL_AGENTS
	dump_agentsCPU("initial agents on CPU", agCPU, 1, 0);
#endif

	// allocate the structure to hold the results
	RESULTS *r = newResults(g_p);

	unsigned warmupTimer;
	unsigned learnTimer;
	unsigned benchmarkTimer;
	CREATE_TIMER(&warmupTimer);
	CREATE_TIMER(&learnTimer);
	CREATE_TIMER(&benchmarkTimer);
	
	// lastWinner is the agent in first place after each learning session
	unsigned lastWinner = 0;

	// warmup vs. random agent
	START_TIMER(warmupTimer);
	if (g_p.warmup_length > 0) {
		printf("warm-up versus RAND\n");
//		dump_agentsCPU("prior to warmup", agCPU, 0, 0);
		for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
			unsigned save_num_pieces = g_p.num_pieces;
			printf("agent %d learning against RAND ... ", iAg);
			for (int p = 1; p <= save_num_pieces; p++) {
				g_p.num_pieces = p;
				printf(" %d ... ", p);
				auto_learn(agCPU, iAg, NULL, g_p.num_pieces, g_p.warmup_length, g_p.max_turns);
			}
			printf(" done\n");
			g_p.num_pieces = save_num_pieces;
		}
	}

	STOP_TIMER(warmupTimer, "CPU warm-up");

	START_TIMER(benchmarkTimer);
	PAUSE_TIMER(benchmarkTimer);
	
	START_TIMER(learnTimer);
	PAUSE_TIMER(learnTimer);
	
	for (int iSession = 0; iSession < g_p.num_sessions; iSession++) {
		RESUME_TIMER(learnTimer);	
		// copy the current weights to the saved_wgts area
		memcpy(agCPU->saved_wgts, agCPU->wgts, g_p.num_agents * g_p.wgts_stride * sizeof(float));
		PAUSE_TIMER(learnTimer);
		
		printf("\n********** Session %d **********\n", iSession);
//		printf("g_p.num_hidden is %d\n", g_p.num_hidden);
		// run a round-robin learning session
		for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
			RESUME_TIMER(learnTimer);
			unsigned iStand = iSession * g_p.num_agents + iAg;
			r->standings[iStand].agent= iAg;
			r->standings[iStand].games = 0;
			r->standings[iStand].wins = 0;
			r->standings[iStand].losses = 0;
			r->vsChamp[iStand].agent= iAg;
			r->vsChamp[iStand].games = 0;
			r->vsChamp[iStand].wins = 0;
			r->vsChamp[iStand].losses = 0;

//			for (int iOp = 0; iOp < g_p.num_agents; iOp++) {
//				unsigned xOp = (iAg + iOp) % g_p.num_agents;

			// compete against the top half of the agents from previous standings
			for (int iOp = 0; iOp < g_p.num_opponents; iOp++){
//			for (int iOp = 0; iOp < 1; iOp++){
				unsigned xOp = iOp;
				if (iSession > 0) xOp = r->standings[(iSession-1) * g_p.num_agents + iOp].agent;

//				printf("\n\n>>>>> new matchup >>>>> (%d vs %d)\n", iAg, xOp);

				WON_LOSS wl = auto_learn(agCPU, iAg, agCPU->saved_wgts + xOp * g_p.wgts_stride, g_p.num_pieces, g_p.episode_length, g_p.max_turns);
//				printf("g_p.num_hidden is %d\n", g_p.num_hidden);
				r->standings[iStand].games += wl.games;
				r->standings[iStand].wins += wl.wins;
				r->standings[iStand].losses += wl.losses;
			}
			PAUSE_TIMER(learnTimer);
			// just compete against the champ
//			WON_LOSS wl = auto_learn(agCPU, iAg, champ_wgts, g_p.num_pieces, g_p.episode_length, g_p.max_turns, g_p.num_hidden);
//			r->standings[iStand].games += wl.games;
//			r->standings[iStand].wins += wl.wins;
//			r->standings[iStand].losses += wl.losses;

			RESUME_TIMER(benchmarkTimer);
			// compete against the champ as a benchmark
			if (g_p.benchmark_games > 0 && 0 == ((1+iSession) % g_p.benchmark_freq)) {
				r->vsChamp[iStand] = compete(agCPU->wgts + iAg * g_p.wgts_stride, NULL, champ_wgts, NULL, g_p.num_pieces, g_p.benchmark_games, g_p.max_turns, 0, g_p.num_hidden, agCPU->seeds + iAg, g_p.num_agents);
			}
			PAUSE_TIMER(benchmarkTimer);
		}
		RESUME_TIMER(learnTimer);
		
		// sort and print the standings
		print_standings(r->standings + iSession * g_p.num_agents, r->vsChamp + iSession * g_p.num_agents);
		
		// remember the winner after this session
		lastWinner = r->standings[iSession * g_p.num_agents + 0].agent;
		PAUSE_TIMER(learnTimer);
	}
	
	STOP_TIMER(learnTimer, "CPU learning");
	STOP_TIMER(benchmarkTimer, "CPU benchmark");
	
	// as a final test, see if the last winner can beat the champ over 1000 games
//	WON_LOSS wlChamp = compete(agCPU->wgts + lastWinner * g_p.wgts_stride, "CHALLENGER", champ_wgts, "CHAMP", g_p.num_pieces, 1000, g_p.max_turns, 0, g_p.num_hidden, agCPU->seeds + lastWinner, g_p.num_agents);
//	printf("CHALLENGER v CHAMP  G: %d  W: %d  L: %d    %+4d   ", wlChamp.games, wlChamp.wins, wlChamp.losses, wlChamp.wins - wlChamp.losses);
//
//	if (wlChamp.wins < wlChamp.losses) {
//		printf("CHAMP wins\n");
//	}else {
//		printf("CHALLENGER beat CHAMP!!!\n");
//	}
#ifdef DUMP_FINAL_AGENTS_CPU
	dump_agentsCPU("agents on CPU, after learning", agCPU, 1, DUMP_SAVED_WGTS);
#endif
	r->iBest = lastWinner;
	return r;
}


#pragma mark -
#pragma mark GPU helpers

// Copy the opponent grid from the device and print it out
void dump_opponent_array()
{
	unsigned *opgrid = host_copyui(g_p.d_opgrid, g_p.num_opponents * g_p.segs_per_session);
	printf("opponents in g_p.d_opgrid:\n");
	for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
		printf("[Segment%3d] ", iSeg);
		for (int iOp = 0; iOp < g_p.num_opponents; iOp++){
			printf(" %3d ", opgrid[iSeg * g_p.num_opponents + iOp]);
		}
		printf("\n");
	}
	free(opgrid);
}

// Set wgts to random values based on init_wgt_min and init_wgt_max
__device__ void randomize_wgts(float *wgts, unsigned *s_seeds, unsigned stride)
{
	unsigned idx = threadIdx.x;
	wgts[idx] = dc_init_wgt_min + (dc_init_wgt_max - dc_init_wgt_min) * RandUniform(s_seeds + idx, stride);
}

// Sum the values for a board, non-destructively
// The total is stored in s_count[0]
// Size of s_count must be at least dc_half_board_size
// *** Requires half_board_size threads ***
__device__ void count_board_pieces(unsigned *s_board, unsigned *s_count)
{
	unsigned idx = threadIdx.x;	
	unsigned half = dc_half_board_size;
	if (idx < half) s_count[idx] = s_board[idx];
	if (idx + half < dc_board_size) s_count[idx] += s_board[idx + half];
	__syncthreads();
	while (0 < (half >>= 1)) {
		if (idx < half) {
			s_count[idx] += s_count[idx + half];
		}
		__syncthreads();
	}
}

// Sum up the values for a board, leaving the total in s_board[0]
// Destroys values in the board.
// *** Requires half_board_size threads ***
__device__ void reduce_board(float *s_board)
{
	unsigned idx = threadIdx.x;
	unsigned half = dc_half_board_size;
	if (idx + half < dc_board_size) s_board[idx] += s_board[idx + half];
	__syncthreads();
	while (0 < (half >>= 1)) {
		if (idx < half) s_board[idx] += s_board[idx + half];
		__syncthreads();
	}
}

// Sum the values for the hidden nodes.
// Destructive.
// *** Requires half_hidden threads ***
__device__ void reduce_hidden(float *s_hidden)
{
	unsigned idx = threadIdx.x;
	unsigned half = dc_half_hidden;
	if (idx + half < dc_num_hidden) s_hidden[idx] += s_hidden[idx + half];
	__syncthreads();
	while (0 < (half >>= 1)) {
		if (idx < half) s_hidden[idx] += s_hidden[idx + half];
		__syncthreads();
	}
}

// Fill in the board with random pieces (uses dc_board_size and dc_num_pieces)
//		s_board points to the board for this agent
//		s_temp is a temporary area of size at least board_size/2
//		idx is the thread number which represents the cell, must range from 0 to (dc_board_size - 1)
//		seeds points to 1st seed for this agent
//		stride is the stride of seeds
// *** Requires board_size threads ***
__device__ void random_board(unsigned *s_board, unsigned *s_temp, unsigned *seeds, unsigned stride)
{
	unsigned idx = threadIdx.x;
	
	// randomly add pieces to cells using probability dc_piece_ratio
	if (idx < dc_board_size) {
		if (dc_piece_ratioX > RandUniform(seeds + idx, stride)) s_board[idx] = 1;
		else s_board[idx] = 0;
	}
	__syncthreads();
	
	count_board_pieces(s_board, s_temp);
	// s_temp[0] now contains the number of 1's
	
	// use thread 0 to add/remove pieces to get the correct total
	if (idx == 0) {
		unsigned r;
		while (s_temp[0] > dc_num_pieces) {
			r = rand_num(dc_board_size, seeds + idx, stride);
			if (s_board[r] == 1) { s_board[r] = 0; --s_temp[0];}
		}
		while (s_temp[0] < dc_num_pieces) {
			r = rand_num(dc_board_size, seeds + idx, stride);
			if (s_board[r] == 0) { s_board[r] = 1; ++s_temp[0];}
		}
	}
	__syncthreads();
}

// Create a random board, but avoid the cells occupied in s_mask
// *** Requires board_size threads ***
__device__ void random_board_masked(unsigned *s_board, unsigned *s_mask, unsigned *s_temp, unsigned *seeds, unsigned stride)
{
	unsigned idx = threadIdx.x;
	
	if (idx < dc_board_size) {
		if (dc_piece_ratioO > RandUniform(seeds + idx, stride) && !s_mask[idx]) s_board[idx] = 1;
		else s_board[idx] = 0;
	}
	__syncthreads();
	
	count_board_pieces(s_board, s_temp);
	// s_temp[0] now contains the number of 1's
	
	if (idx == 0) {
		unsigned r;
		while (s_temp[0] > dc_num_pieces) {
			r = rand_num(dc_board_size, seeds + idx, stride);
			if (s_board[r] == 1) { s_board[r] = 0; --s_temp[0];}
		}
		while (s_temp[0] < dc_num_pieces) {
			r = rand_num(dc_board_size, seeds + idx, stride);
			if (s_board[r] == 0 && s_mask[r] == 0){ s_board[r] = 1; ++s_temp[0];}
		}
	}
	__syncthreads();
}

// Create a random starting state
// *** Requires board_size threads ***
__device__ void random_stateGPU(unsigned *s_state, float *s_temp, unsigned *s_seeds, unsigned stride)
{
	random_board(s_state, (unsigned *)s_temp, s_seeds, dc_board_size);
	random_board_masked(s_state + dc_board_size, s_state, (unsigned *)s_temp, s_seeds, dc_board_size);
}

// Calculate the reward for a given state, setting the terminal flag, ps_terminal,
// and storing the reward in ps_reward.
// *** Requires half_board_size threads ***
__device__ void rewardGPU(unsigned *s_state, unsigned *s_temp, unsigned *ps_terminal, float *ps_reward)
{
	unsigned idx = threadIdx.x;
	if (idx == 0){ *ps_terminal = 0; *ps_reward = 0.0f; }
	__syncthreads();
	
	count_board_pieces(O_BOARDGPU(s_state), s_temp);
	if (idx == 0 && 0 == s_temp[0]) {
		*ps_terminal = 1;
		*ps_reward = REWARD_WIN;
	}
	__syncthreads();
	count_board_pieces(X_BOARDGPU(s_state), s_temp);
	if (idx == 0 && 0 == s_temp[0]) {
		*ps_terminal = 1;
		*ps_reward = REWARD_LOSS;
	}
	__syncthreads();
}

// Switch X and O pieces of a state
// *** Requires board_size threads ***
__device__ void switch_sidesGPU(unsigned *s_state)
{
	unsigned idx = threadIdx.x;
	unsigned temp = s_state[idx];
	s_state[idx] = s_state[idx + dc_board_size];
	s_state[idx + dc_board_size] = temp;
}

// Copy weights from global memory to shared memory
//	*** Requires board_size threads ***
__device__ void copy_wgts_to_s(float *g_wgts, float *s_wgts)
{
	unsigned idx = threadIdx.x;
	if (idx < dc_board_size) {
		for (int iH = 0; iH < dc_num_hidden; iH++) {
			S_IXH(s_wgts, iH)[idx] = G_IXH(g_wgts, iH)[idx];
			S_IOH(s_wgts, iH)[idx] = G_IOH(g_wgts, iH)[idx];
		}
	}
		
	if (idx < dc_num_hidden) {
		S_BH(s_wgts)[idx] = G_BH(g_wgts)[idx];
		S_HO(s_wgts)[idx] = G_HO(g_wgts)[idx];
	}

	if (idx == 0) {
		S_BO(s_wgts)[idx] = G_BO(g_wgts)[idx];
	}

//	s_wgts[idx] = g_wgts[idx];
	
	__syncthreads();
}

// Copy weights from shared memory to global memory
//	*** Requires board_size threads ***
__device__ void copy_wgts_to_g(float *s_wgts, float *g_wgts)
{
	unsigned idx = threadIdx.x;
	if (idx < dc_board_size) {
		for (int iH = 0; iH < dc_num_hidden; iH++) {
			G_IXH(g_wgts, iH)[idx] = S_IXH(s_wgts, iH)[idx];
			G_IOH(g_wgts, iH)[idx] = S_IOH(s_wgts, iH)[idx];
		}
	}
	if (idx < dc_num_hidden) {
		G_BH(g_wgts)[idx] = S_BH(s_wgts)[idx];
		G_HO(g_wgts)[idx] = S_HO(s_wgts)[idx];
	}

	if (idx == 0) {
		G_BO(g_wgts)[idx] = S_BO(s_wgts)[idx];
	}

//	g_wgts[idx] = s_wgts[idx];
	__syncthreads();
}

// calculate the difference between the old weights (in global memory) and the new weights
// (in shared memory) and store the difference in global memory (in compact format) at delta_wgts
__device__ void calc_delta_wgts(float *new_wgts, float *old_wgts, float *delta_wgts)
{
	unsigned idx = threadIdx.x;
	if (idx < dc_board_size) {
		for (int iH = 0; iH < dc_num_hidden; iH++) {
			S_IXH(delta_wgts, iH)[idx] = S_IXH(new_wgts, iH)[idx] - G_IXH(old_wgts, iH)[idx];
			S_IOH(delta_wgts, iH)[idx] = S_IOH(new_wgts, iH)[idx] - G_IOH(old_wgts, iH)[idx];
		}
	}
	
	if (idx < dc_num_hidden) {
		S_BH(delta_wgts)[idx] = S_BH(new_wgts)[idx] - G_BH(old_wgts)[idx];
		S_HO(delta_wgts)[idx] = S_HO(new_wgts)[idx] - G_HO(old_wgts)[idx];
	}

	if (idx == 0) {
		S_BO(delta_wgts)[idx] = S_BO(new_wgts)[idx] - G_BO(old_wgts)[idx];
	}
	__syncthreads();
}


#pragma mark -
#pragma mark GPU code

// Calcualte the value for a state s using the specified weights,
// storing hidden activation in the specified location and storing output value in *ps_V
// *** Requires board_size threads ***
__device__ void val_for_stateGPU(float *s_wgts, unsigned *s_state, float *s_hidden, float *s_out, float *s_temp, unsigned *ps_terminal, float *ps_V)
{
	unsigned idx = threadIdx.x;

	// first check for terminal condition, returning reward if is terminal
	rewardGPU(s_state, (unsigned *)s_temp, ps_terminal, ps_V);
	if(*ps_terminal) return;
	
	// repeat for each hidden node...
	for (int iH = 0; iH < dc_num_hidden; iH++) {

		// add in the X-piece and O-piece contributions
		s_temp[idx] = 0.0f;
		if (X_BOARDGPU(s_state)[idx]) s_temp[idx] += S_IXH(s_wgts, iH)[idx];
		if (O_BOARDGPU(s_state)[idx]) s_temp[idx] += S_IOH(s_wgts, iH)[idx];
		
		// reduce to get the input value for this hidden node (before adding bias)
		reduce_board(s_temp);
		if (idx == 0) s_hidden[iH] = s_temp[0];
		__syncthreads();
	}
	
	// s_hidden now contains the input to the hidden nodes before adding the bias
	if (idx < dc_num_hidden){
		s_hidden[idx] += -1.0f * S_BH(s_wgts)[idx];
		s_hidden[idx] = sigmoid(s_hidden[idx]);
		s_out[idx] = s_hidden[idx] * S_HO(s_wgts)[idx];
	}
	__syncthreads();
	
	reduce_hidden(s_out);
	if (idx == 0) {
		// add in the bias to the output, then apply sigmoid
		s_out[0] += -1.0f * S_BO(s_wgts)[0];
		s_out[0] = sigmoid(s_out[0]);
		*ps_V = s_out[0];
	}
	__syncthreads();
}

// Reset trace values to 0.0f
// *** Requires board_size threads ***
__device__ void reset_traceGPU(float *s_e)
{
	unsigned idx = threadIdx.x;
		
	for (int i = 0; i < dc_reps_for_wgts; i++) {
		if (idx + i * dc_board_size < dc_num_wgts) {
			s_e[idx + i * dc_board_size] = 0.0f;
		}
	}
	__syncthreads();
}

// Update the weights based on delta value and eligibility trace
//		(new wgt) = (old wgt) + (alpha * delta * eligibility trace)
// *** Requires board_size threads ***
__device__ void update_wgtsGPU(float s_alpha, float delta, float *s_wgts, float *s_e)
{
	unsigned idx = threadIdx.x;
	
	for (int i = 0; i < dc_reps_for_wgts; i++) {
		if (idx + i * dc_board_size < dc_num_wgts) {
			s_wgts[idx + i * dc_board_size] += s_alpha * delta * s_e[idx + i * dc_board_size];
		}
	}
	__syncthreads();
}

// Update the eligibility trace
//		First, eligibility trace is decayed by a factor of gamma * lambda
//		Next, the eligitity traces are increased using the gradients
// On entry, s_out[0] contains the output activation values,
//           s_hidden contains the activation values for the hidden nodes,
//           s_temp is a working area with of size >= board_size
// *** Requires board_size threads ***
__device__ void update_traceGPU(unsigned *s_state, float *s_wgts, float *s_e, float *s_hidden, float *s_out, float lambda, float *s_temp)
{
	unsigned idx = threadIdx.x;
	
	// decay all existing values
	for (int i = 0; i < dc_reps_for_wgts; i++) {
		if (idx + i * dc_board_size < dc_num_wgts) {
			s_e[idx + i * dc_board_size] *= dc_gamma * lambda;
		}
	}
	__syncthreads();
	
	// update weights from hidden layer to output node
	s_temp[idx] = s_out[0] * (1.0f - s_out[0]);			// store g_prime_i in s_temp[]
	if (idx < dc_num_hidden) {
		if (idx == 0) S_BO(s_e)[0] += -1.0f * s_temp[idx];
		S_HO(s_e)[idx] += s_hidden[idx] * s_temp[idx];
	}
	__syncthreads();
	
	// update weights from input to hidden nodes
	for (int iH = 0; iH < dc_num_hidden; iH++) {
		s_temp[idx] = s_hidden[iH] * (1.0f - s_hidden[iH]) * S_HO(s_wgts)[iH] * s_out[0] * (1.0f - s_out[0]);	// g_prime_j
		if (idx == 0) S_BH(s_e)[iH] += -1.0f * s_temp[idx];
		if (X_BOARDGPU(s_state)[idx]) S_IXH(s_e, iH)[idx] += s_temp[idx];
		if (O_BOARDGPU(s_state)[idx]) S_IOH(s_e, iH)[idx] += s_temp[idx];
	}
	__syncthreads();
}

// choose a random move for X
__device__ void random_moveGPU(unsigned *s_state, unsigned *s_temp, unsigned *ps_terminal, float *ps_V, unsigned *s_seeds, unsigned stride)
{
	unsigned idx = threadIdx.x;
	int iTo;
	int iFrom;
	
	// check for terminal condition
	rewardGPU(s_state, (unsigned *)s_temp, ps_terminal, ps_V);
	if (*ps_terminal) return;

	// use *ps_terminal to flag that we've found a legal move
	*ps_terminal = 0;
	while (!*ps_terminal) {
		// each thread will choose a random move from its cell and then determine if its legal
		if (X_BOARDGPU(s_state)[idx]) {
			unsigned m = rand_num(MAX_MOVES, s_seeds, stride);

#ifdef USE_TEXTURE_FOR_MOVES
			iTo = tex2D(texRef, idx, m);
#else
			iTo = dc_moves[m * dc_board_size + idx];
#endif

			if (iTo >= 0 && !X_BOARDGPU(s_state)[iTo]) {
				// found a possible move, record <from> <to> and <rand_val> into one int in s_temp
				s_temp[idx] = idx | (iTo << dc_board_bits) | (RandUniformui(s_seeds, stride) << (2*dc_board_bits));
				*ps_terminal = 1;
			}
		}
		__syncthreads();
	}
	
	// do a reduction on s_temp and then decode the random move
	unsigned half = dc_half_board_size;
	if ((idx + half < dc_board_size) && (s_temp[idx] < s_temp[idx + half])) 
		s_temp[idx] = s_temp[idx + half];
	__syncthreads();
	while (0 < (half >>= 1)) {
		if ((idx < half) && (s_temp[idx] < s_temp[idx + half])) s_temp[idx] = s_temp[idx + half];
		__syncthreads();
	}
	
	if(idx == 0){
		// decode the from and to cells
		iFrom = s_temp[0] & ((1 << dc_board_bits) - 1);		// lowest dc_board_bits bits
		iTo = (s_temp[0] >> dc_board_bits) & ((1 << dc_board_bits) -1);	// next dc_board_bits
		
		// do the move
		X_BOARDGPU(s_state)[iFrom] = 0;
		X_BOARDGPU(s_state)[iTo] = 1;
		O_BOARDGPU(s_state)[iTo] = 0;
	}
	__syncthreads();
}

/*
	Choose the next move for X.
	The next move is the move with the highest value for the resulting state.
	The state is modified for the chosen move, and the terminal flag, ps_terminal is set.
	The value for the new state is stored in *ps_V.
	
	*** Requires board_size threads ***
*/
__device__ void choose_moveGPU(unsigned *s_state, float *s_temp, float *s_wgts, float *s_hidden, float *s_out, unsigned *ps_terminal, float *ps_V)
{
	unsigned idx = threadIdx.x;
	
	unsigned noVal = 1;
	float bestVal;
	unsigned iBestFrom;
	unsigned iBestTo;

	// check for terminal condition
	rewardGPU(s_state, (unsigned *)s_temp, ps_terminal, ps_V);
	if (*ps_terminal) return;
	
	for (int iFrom = 0; iFrom < dc_board_size; iFrom++) {
		if (X_BOARDGPU(s_state)[iFrom]) {
			for (int m = 0; m < MAX_MOVES; m++) {

#ifdef USE_TEXTURE_FOR_MOVES
				int iTo = tex2D(texRef, iFrom, m);
#else
				int iTo = dc_moves[m * dc_board_size + iFrom];
#endif

				if (iTo >= 0 && !X_BOARDGPU(s_state)[iTo]){
					// found a possible move, modify the board and calculate the value
					unsigned oPiece;
					if (idx == 0){
						oPiece = O_BOARDGPU(s_state)[iTo];	// remember if there was an O piece here
						X_BOARDGPU(s_state)[iFrom] = 0;
						X_BOARDGPU(s_state)[iTo] = 1;
						O_BOARDGPU(s_state)[iTo] = 0;
					}
					__syncthreads();
					
					val_for_stateGPU(s_wgts, s_state, s_hidden, s_out, s_temp, ps_terminal, ps_V);
					
					if (idx == 0) {
						if (noVal || *ps_V > bestVal) {
							// record the best move so far
							iBestFrom = iFrom;
							iBestTo = iTo;
							bestVal = *ps_V;
							noVal = 0;
						}
						// restore the state
						X_BOARDGPU(s_state)[iFrom] = 1;
						X_BOARDGPU(s_state)[iTo] = 0;
						O_BOARDGPU(s_state)[iTo] = oPiece;
					}
					__syncthreads();
				}
			}
		}
	}
	
	// do the best move
	if (idx == 0) {
		X_BOARDGPU(s_state)[iBestFrom] = 0;
		X_BOARDGPU(s_state)[iBestTo] = 1;
		O_BOARDGPU(s_state)[iBestTo] = 0;
	}
	__syncthreads();
	
	// recalculate value to set the s_out and s_hidden activation values
	val_for_stateGPU(s_wgts, s_state, s_hidden, s_out, s_temp, ps_terminal, ps_V);
}


/*
	Player O makes a move based on the highest resulting board value from O's perspective.
	X and O sides of the state are switched and then choose_moveGPU is used to select the move
	Sets terminal flag and reward values.
	
	*** Requires board_size threads ***
*/
__device__ void take_actionGPU(unsigned *s_state, float *s_temp, float *s_owgts, float *s_hidden, float *s_out, unsigned *ps_terminal, float *s_ophidden, float *ps_reward)
{
	rewardGPU(s_state, (unsigned *)s_temp, ps_terminal, ps_reward);
	if (!*ps_terminal) {
		switch_sidesGPU(s_state);
		choose_moveGPU(s_state, s_temp, s_owgts, s_ophidden, s_out, ps_terminal, ps_reward);
		switch_sidesGPU(s_state);
		rewardGPU(s_state, (unsigned *)s_temp, ps_terminal, ps_reward);
	}
	__syncthreads();
}

// Player O makes a random move
__device__ void take_random_actionGPU(unsigned *s_state, float *s_temp, float *s_owgts, float *s_hidden, float *s_out, unsigned *ps_terminal, float *s_ophidden, float *ps_reward, unsigned *s_seeds, unsigned s_stride)
{
	randomize_wgts(s_owgts, s_seeds, s_stride);
	take_actionGPU(s_state, s_temp, s_owgts, s_hidden, s_out, ps_terminal, s_ophidden, ps_reward);
}

// ***TODO convert this to a simple kernel
// update the opponent array, g_p.opgrid, using the specified method
void update_opgrid(enum OPPONENT_METHODS method, WON_LOSS *lastStandings, unsigned param)
{
	switch (method) {
		// opgrid is be segment and opponent, so can't code the self agent number
		case OM_SELF:	// compete against self (param not used)
			printf("*** ERROR *** OM_SELF opponent selection method not implemented\n");
			break;
			
		case OM_FIXED1:	// Compete against fixed group of agents [0, num_opponents) in parallel
						// Same group is repeated for every segment.
						// Number of different opponents for an agent = num_opponents
						// eg:	seg1: 0 1 2 3
						//		seg2: 0 1 2 3
						//		seg3: 0 1 2 3
						//			... etc
			printf("updating opponents: OM_FIXED1\n");
			for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
				for (int iOp = 0; iOp < g_p.num_opponents; iOp++) {
					g_p.opgrid[iSeg * g_p.num_opponents + iOp] = iOp;
				}
			}
			break;
			
		case OM_FIXED2:	// compete against fixed group of agents, one agent per segment,
						// same agent repeated for parallel learning within a segment
						// Number of different agents = segs_per_session
						// eg:	seg1: 0 0 0 0
						//		seg2: 1 1 1 1
						//		seg3: 2 2 2 2
						//			... etc
			printf("updating opponents: OM_FIXED2\n");
			for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
				for (int iOp = 0; iOp < g_p.num_opponents; iOp++) {
					g_p.opgrid[iSeg * g_p.num_opponents + iOp] = iSeg % g_p.num_agents;
				}
			}
			break;
			
		case OM_BEST1:	// compete against the best agents from the previous standings
						// in a pattern similar to fixed1, but chosing agents based on the standings
						// eg:	seg1: 1st place		2nd place	3rd place	4th place
						//		seg2: 1st place		2nd place	3rd place	4th place
						//		seg3: 1st place		2nd place	3rd place	4th place
						//			... etc
			if (!lastStandings) return update_opgrid(OM_FIXED1, NULL, param);
			printf("updating opponents: OM_BEST1\n");
			for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
				for (int iOp = 0; iOp < g_p.num_opponents; iOp++) {
					g_p.opgrid[iSeg * g_p.num_opponents + iOp] = lastStandings[iOp].agent;
				}
			}
			break;
			
		case OM_BEST2:	// compete against the best agents from the previous standings
						// in a pattern similar to fixed2, but chosing agents based on the standings
						//(param is pointer to lastStandings)
						// eg:	seg1: 1st place		1st place	1st place
						//		seg2: 2nd place		2nd place	2nd place
						//		seg3: 3rd place		3rd place	3rd place
						//			... etc
			if (!lastStandings) return update_opgrid(OM_FIXED2, NULL, param);
			printf("updating opponents: OM_BEST2\n");
			for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
				for (int iOp = 0; iOp < g_p.num_opponents; iOp++) {
					g_p.opgrid[iSeg * g_p.num_opponents + iOp] = lastStandings[iSeg % g_p.num_agents].agent;
				}
			}
			break;
			
		case OM_ONE:	// compete against the one specified opponent (param is opponent agent number)
			printf("updating opponents: OM_ONE\n");
			for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {
				for (int iOp = 0; iOp < g_p.num_opponents; iOp++) {
					g_p.opgrid[iSeg * g_p.num_opponents + iOp] = param;
				}
			}
			break;
			
		default:
			printf("*** ERROR *** Unknown opponent selection method\n");
			break;
	}
	
	// finally, copy the opponent grid to the device
	CUDA_SAFE_CALL(cudaMemcpy(g_p.d_opgrid, g_p.opgrid, g_p.num_opponents * g_p.segs_per_session * sizeof(unsigned), cudaMemcpyHostToDevice));
	
	printf("just stored opponent array on device:\n");
	dump_opponent_array();
}

// sum up the won-loss results for each agent 
// threadIdx.x is the opponent number
// blockIdx.x is the agent number
__global__ void reduce_wl_kernel(WON_LOSS *wl, WON_LOSS *wl_tot, unsigned num_ops, unsigned half_num_ops)
{
	unsigned idx = threadIdx.x;
	unsigned iAgent = blockIdx.x;
	
	extern __shared__ unsigned s_games[];		// num_ops
	unsigned *s_wins = s_games + num_ops;		// num_ops
	unsigned *s_losses = s_wins + num_ops;		// num_ops
	
	// copy won-loss data to shared memeory
//	s_games[idx] = wl[iAgent * dc_num_opponents + idx].games;
//	s_wins[idx] = wl[iAgent * dc_num_opponents + idx].wins;
//	s_losses[idx] = wl[iAgent * dc_num_opponents + idx].losses;
	s_games[idx] = wl[iAgent * num_ops + idx].games;
	s_wins[idx] = wl[iAgent * num_ops + idx].wins;
	s_losses[idx] = wl[iAgent * num_ops + idx].losses;
	__syncthreads();
	
	// reduce the values in shared memory
//	unsigned half = dc_half_opponents;
	unsigned half = half_num_ops;
	if (half > 0 && (idx + half) < num_ops) {
		s_games[idx] += s_games[idx + half];
		s_wins[idx] += s_wins[idx + half];
		s_losses[idx] += s_losses[idx + half];
	}
	__syncthreads();
	while (0 < (half >>= 1)) {
		if (idx < half){
			s_games[idx] += s_games[idx + half];
			s_wins[idx] += s_wins[idx + half];
			s_losses[idx] += s_losses[idx + half];
		}
		__syncthreads();
	}
	
	// copy the totals out to global memory
	if (idx == 0){
		wl_tot[iAgent].agent = iAgent;
		wl_tot[iAgent].games = s_games[0];
		wl_tot[iAgent].wins = s_wins[0];
		wl_tot[iAgent].losses = s_losses[0];
	}
}

// calculate the total delta for the agent and update their global weights
// total x dimension is the index into the delta and wgts
// old_wgts and new_wgts are in global weight format
// delta_wgts are in compact format
__global__ void share_delta_kernel(float *old_wgts, float *delta_wgts, float *new_wgts)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned iAgent = blockIdx.y;
	
	extern __shared__ float s_wgts[];		// num_wgts
	
	// first fill the new wgts in shared memeory with the old wgt values
	copy_wgts_to_s(old_wgts + iAgent * dc_wgts_stride, s_wgts);

	// next add in all the deltas from every opponent
	for (int iOpponent = 0; iOpponent < dc_num_opponents; iOpponent++) {
		s_wgts[idx] += delta_wgts[iAgent * dc_num_opponents * dc_num_wgts + dc_num_wgts * iOpponent + idx];
	}
	
	// copy the wgts back to global memory
	copy_wgts_to_g(s_wgts, new_wgts + iAgent * dc_wgts_stride);
}


// copy all weights to the saved weights area
// total x dimension determines the index within dc_wgts_stride for this thread
// The block's y dimension is the agent number
__global__ void copy_wgts_kernel(float *wgts, float *saved_wgts)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned iAgent = blockIdx.y;
	unsigned iGlobal = iAgent * dc_wgts_stride + idx;
	if (idx < dc_wgts_stride) {
		saved_wgts[iGlobal] = wgts[iGlobal];
	}
}

/*
	Compete (without learning) against a benchmark opponent, or against other agents.
	If round_robin flag is on, the opwgts points to agent weights and the iOpponet value indexes
	into it to obtain the opponent weights for the competition.
	If round_robin flag is off, then opwgts is pointing to the single benchmark opponent's weights.
*/
__global__ void compete_kernel(unsigned *seeds, float *wgts, float *opwgts, WON_LOSS *wl, unsigned round_robin)
{
	unsigned idx = threadIdx.x;
	unsigned iAgent = blockIdx.x;
	unsigned iOpponent = blockIdx.y;	// all opponents are the same single benchmark opponent, but may vary in the future

	__shared__ float s_rand;
	__shared__ float s_reward;
	__shared__ unsigned s_terminal;
	__shared__ unsigned s_stats[3];		// games, wins, and losses
	__shared__ unsigned s_tempui;
	__shared__ float s_tempf;
	
	// dynamic shared memory
	extern __shared__ unsigned s_seeds[];				// 4 * dc_board_size
	unsigned *s_state = s_seeds + 4 * dc_board_size;	// dc_state_size
	float *s_temp = (float *)(s_state + dc_state_size);	// dc_board_size
	float *s_hidden = s_temp + dc_board_size;			// dc_num_hidden
	float *s_out = s_hidden + dc_num_hidden;			// dc_num_hidden
	float *s_ophidden = s_out + dc_num_hidden;			// dc_num_hidden
	float *s_wgts = s_ophidden + dc_num_hidden;			// dc_num_wgts
	float *s_opwgts = s_wgts + dc_num_wgts;				// dc_num_wgts
	
	// copy values to shared memory and initialize
	if (idx == 0) {
		s_stats[0] = 0;
		s_stats[1] = 0;
		s_stats[2] = 0;
	}
	__syncthreads();
	
	// Vary the seeds using the opponent index in blockIdx.y so won't get the same results in
	// parallel runs vs. samve opponent.
	s_seeds[idx] = seeds[iAgent * 4 * dc_board_size + idx]+ (blockIdx.y << 4);
	s_seeds[idx + dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + dc_board_size]+ (blockIdx.y << 4);
	s_seeds[idx + 2*dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + 2*dc_board_size]+ (blockIdx.y << 4);
	s_seeds[idx + 3*dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + 3*dc_board_size]+ (blockIdx.y << 4);
	
	copy_wgts_to_s(wgts + iAgent * dc_wgts_stride, s_wgts);
	copy_wgts_to_s(round_robin ? (opwgts + iOpponent * dc_wgts_stride) : opwgts, s_opwgts);

	unsigned turn = 0;

	// set up initial random state
	random_stateGPU(s_state, s_temp, s_seeds, dc_board_size);

	if (idx == 0){
		s_rand = RandUniform(s_seeds, dc_board_size);
		s_reward = 0.0f;
	}
	__syncthreads();

	if (s_rand < 0.50f){
		take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward);
		++turn;
	}
	__syncthreads();
	choose_moveGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, &s_tempf);
	
	while (s_stats[0] < dc_benchmark_games_per_op) {
		take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward);
		++turn;
		
		if (s_terminal || turn == dc_max_turns) {
			turn = 0;
			if (idx == 0) {
				++s_stats[0];
				if (s_terminal) {
					if (s_reward > 0.50f) ++s_stats[1];
					else ++s_stats[2];
				}
				s_rand = RandUniform(s_seeds, dc_board_size);
			}
			__syncthreads();
			random_stateGPU(s_state, s_temp, s_seeds, dc_board_size);
			if (s_rand < 0.50f) {
				take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_tempui, s_ophidden, &s_tempf);
				++turn;
			}
		}
		choose_moveGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, &s_tempf);
	}
	
	if (wl){
		if (idx == 0) wl[iAgent * dc_benchmark_ops + iOpponent].agent = iAgent;
		if (idx < 3) {
			*(1 + idx + (unsigned *)(wl + iAgent * dc_benchmark_ops + iOpponent)) = s_stats[idx];
		}
	}

}

/*
	Run an episode of learning against a static opponent
	
	Arguments are the global values for all agents for seeds, wgts, and eligibility trace
	saved_wgts contains the weights for the opponent

	blockIdx.x is the agent number for the learner
	blockIdx.y is the index into d_ops to determine the opponent.
	
	*** Requires board_size threads ***
*/
__global__ void old_learn_kernel(unsigned *seeds, float *wgts, float *e, float *saved_wgts, WON_LOSS *wl, float *delta_wgts, unsigned *d_ops, unsigned reset_stats, unsigned *max_turns)
{
	unsigned idx = threadIdx.x;
	unsigned iAgent = blockIdx.x;
	
	// blockIdx.y is the index within the opponent array, d_ops
	// iOpponent is the agent number of the opponenent for this episode
	unsigned iOpponent = d_ops[blockIdx.y];
	
	// static shared memory
	__shared__ float s_rand;
	__shared__ float s_reward;
	__shared__ float s_lambda;
	__shared__ float s_alpha;
	__shared__ float s_V;
	__shared__ float s_V_prime;
	__shared__ float s_delta;
	__shared__ unsigned s_terminal;
	__shared__ unsigned s_stats[3];		// games, wins, and losses
	__shared__ unsigned s_tempui;
	__shared__ float s_tempf;

	// dynamic shared memory								------ size ------
	extern __shared__ unsigned s_seeds[];					// 4 * dc_board_size
	unsigned *s_state = s_seeds + 4 * dc_board_size;		// dc_state_size
	float *s_temp = (float *)(s_state + dc_state_size);		// dc_board_size
	float *s_hidden = s_temp + dc_board_size;				// dc_num_hidden
	float *s_out = s_hidden + dc_num_hidden;				// dc_num_hidden
	float *s_ophidden = s_out + dc_num_hidden;				// dc_num_hidden
	float *s_wgts = s_ophidden + dc_num_hidden;				// dc_num_wgts
	float *s_e = s_wgts + dc_num_wgts;						// dc_num_wgts
	float *s_opwgts = s_e + dc_num_wgts;					// dc_num_wgts
	
	// copy individual values to shared memory and initialize
	if (idx == 0){
		s_lambda = dc_ag.lambda[iAgent];
		s_alpha = dc_ag.alpha[iAgent];
	}
	// reset the stats or copy from global memory so they can be updated
	if (idx < 3) {
		if (reset_stats) s_stats[idx] = 0;
		else s_stats[idx] = *(1 + idx + (unsigned *)(wl + iAgent * dc_num_opponents + blockIdx.y));
	}
	__syncthreads();
	
	// add a value to the seeds so different random numbers for each opponent
	s_seeds[idx] = seeds[iAgent * 4 * dc_board_size + idx] + (blockIdx.y << 4);
	s_seeds[idx + dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + dc_board_size]+ (blockIdx.y << 4);
	s_seeds[idx + 2*dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + 2*dc_board_size]+ (blockIdx.y << 4);
	s_seeds[idx + 3*dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + 3*dc_board_size]+ (blockIdx.y << 4);

	// copy agent weights to shared memory and reset eligibility trace
	copy_wgts_to_s(wgts + iAgent * dc_wgts_stride, s_wgts);
	reset_traceGPU(s_e);

	// copy opponent weights to shared memory, if given
	if (saved_wgts) copy_wgts_to_s(saved_wgts + iOpponent * dc_wgts_stride, s_opwgts);

	unsigned turn = 0;
	unsigned total_turns = 0;
	
	random_stateGPU(s_state, s_temp, s_seeds, dc_board_size);

	if (idx == 0){
		s_rand = RandUniform(s_seeds, dc_board_size);
		s_reward = 0.0f;
	}
	__syncthreads();

	if (s_rand < 0.50f){
		if (saved_wgts) take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward);
		else take_random_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward, s_seeds, dc_board_size);
		++turn;
	}
	__syncthreads();
	
	choose_moveGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, &s_V);
	update_traceGPU(s_state, s_wgts, s_e, s_hidden, s_out, s_lambda, s_temp);
	
	while (total_turns++ < dc_episode_length) {
		if (saved_wgts) take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward);
		else take_random_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward, s_seeds, dc_board_size);
		++turn;
		
		if (s_terminal || (turn == max_turns[iAgent])) {
			turn = 0;
			if (idx == 0) {
				++s_stats[0];
				if (s_terminal){
					if (s_reward > 0.50f) ++s_stats[1];
					else ++s_stats[2];
				}
				s_rand = RandUniform(s_seeds, dc_board_size);
			}
			__syncthreads();			

			random_stateGPU(s_state, s_temp, s_seeds, dc_board_size);

			if (s_rand < 0.50f) {
				if (saved_wgts) take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_tempui, s_ophidden, &s_tempf);
				else take_actionGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, s_ophidden, &s_tempf);
				++turn;
			}
		}
		
		choose_moveGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, &s_V_prime);

		if (idx == 0){
			s_delta = s_reward + (s_terminal ? 0.0f : (dc_gamma * s_V_prime)) - s_V;
		}
		__syncthreads();
		
		update_wgtsGPU(s_alpha, s_delta, s_wgts, s_e);
		if (s_terminal) reset_traceGPU(s_e);
		update_traceGPU(s_state, s_wgts, s_e, s_hidden, s_out, s_lambda, s_temp);

		if (idx == 0) s_V = s_V_prime;
		__syncthreads();
	}

	// copy values back to global memory (states are for information only)
	dc_ag.states[iAgent * dc_state_size + idx] = s_state[idx];
	dc_ag.states[iAgent * dc_state_size + idx + dc_board_size] = s_state[idx + dc_board_size];
	
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx] = s_seeds[idx];
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx + dc_board_size] = s_seeds[idx + dc_board_size];
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx + 2*dc_board_size] = s_seeds[idx + 2*dc_board_size];
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx + 3*dc_board_size] = s_seeds[idx + 3*dc_board_size];

//	copy_wgts_to_g(s_wgts, wgts + iAgent * dc_wgts_stride);		// not needed -- use delta_wgts to get new wgts
//	copy_wgts_to_g(s_e, e + iAgent * dc_wgts_stride);			// for information only

	// caclulate delta_wgts and store in global memory
	calc_delta_wgts(s_wgts, saved_wgts + iAgent * dc_wgts_stride, delta_wgts + iAgent * dc_num_wgts * dc_num_opponents + iOpponent * dc_num_wgts);

	// accumulate the won/loss record in global memory 
	if (wl) {
		if (idx == 0) wl[iAgent*dc_num_opponents + iOpponent].agent = iAgent;
		if (idx < 3){
			*(1 + idx + (unsigned *)(wl + iAgent * dc_num_opponents + blockIdx.y)) = s_stats[idx];
		}
	}
}


// calculate the amount of dynamic shared memory needed
unsigned dynamic_shared_mem()
{
	unsigned count = 4 * g_p.board_size;	// s_seeds
	count += g_p.state_size;				// s_state
	count += g_p.board_size;				// s_temp
	count += g_p.num_hidden;				// s_hidden
	count += g_p.num_hidden;				// s_out
	count += g_p.num_hidden;				// s_ophidden
	count += g_p.num_wgts;					// s_wgts
	count += g_p.num_wgts;					// s_e
	count += g_p.num_wgts;					// s_opwgts
//	printf("%d elements in shared memory, total of %d bytes\n", count, count * 4);
	return sizeof(unsigned) * count;
}

RESULTS *runGPU(AGENT *agGPU, float *champ_wgts)
{
	printf("\n\n-----------------------------------------------------------\nrunning on GPU...\n");

#ifdef DUMP_INITIAL_AGENTS
	dump_agentsGPU("initial agents on GPU", agGPU, 1, 0);
#endif

	// copy champ_wgts to device
	float *d_champ_wgts = device_copyf(champ_wgts, g_p.wgts_stride);
	
	// allocate structure for results of learning and benchmarking on device
	RESULTS *rGPU = newResultsGPU(g_p);
	
	// allocate temporary arrays to hold the results of learning and benchmarking of last session
	WON_LOSS *lastStandings = (WON_LOSS *)malloc(g_p.num_agents * sizeof(WON_LOSS));
	WON_LOSS *lastVsChamp = (WON_LOSS *)malloc(g_p.num_agents * sizeof(WON_LOSS));
	
	// setup timers
//	unsigned gpuWarmupTimer;
	unsigned gpuCopyWgtsTimer;
	unsigned gpuShareDeltaTimer;
	unsigned gpuLearnTimer;
	unsigned gpuReduceWonLossTimer;
	unsigned gpuCompeteTimer;
	unsigned gpuStandingsTimer;
//	CREATE_TIMER(&gpuWarmupTimer);
	CREATE_TIMER(&gpuCopyWgtsTimer);
	CREATE_TIMER(&gpuShareDeltaTimer);
	CREATE_TIMER(&gpuLearnTimer);
	CREATE_TIMER(&gpuReduceWonLossTimer);
	CREATE_TIMER(&gpuCompeteTimer);
	CREATE_TIMER(&gpuStandingsTimer);
	
	dim3 blockDim(g_p.board_size);
	dim3 gridDim(g_p.num_agents);
	
	dim3 blockDim_wgtCopy(g_p.wgts_stride);
	dim3 gridDim_wgtCopy(1, g_p.num_agents);
	if (blockDim_wgtCopy.x > 512) {
		blockDim_wgtCopy.x = 512;
		gridDim_wgtCopy.x = 1 + (g_p.wgts_stride - 1) / 512;
	}
	
	dim3 blockDim_share(g_p.num_wgts);
	dim3 gridDim_share(1, g_p.num_agents);
	if (blockDim_share.x > 512) {
		blockDim_share.x = 512;
		gridDim_share.x = 1 + (g_p.num_wgts - 1) / 512;
	}
	
//	// warm-up vs. random agent
//	START_TIMER(gpuWarmupTimer);
//	if (g_p.warmup_length > 0) {
//		printf("GPU warm-up versus RAND\n");
//		unsigned save_num_pieces = g_p.num_pieces;
//		for (int p = 1; p <= save_num_pieces; p++) {
//			g_p.num_pieces = p;
//			CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_pieces", &g_p.num_pieces, sizeof(unsigned)));
//			printf(" %d ... ", p);
//			PRE_KERNEL("learn_kernel");
//			learn_kernel<<<gridDim, blockDim, dynamic_shared_mem()>>>(agGPU->seeds, agGPU->wgts, agGPU->e, NULL, NULL);
//			POST_KERNEL(learn_kernel);
//		}
//		g_p.num_pieces = save_num_pieces;
//		CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_pieces", &g_p.num_pieces, sizeof(unsigned)));
//		printf(" done\n");
//	}
//	cudaThreadSynchronize();
//	STOP_TIMER(gpuWarmupTimer, "GPU warm-up");
	
	
	START_TIMER(gpuCopyWgtsTimer);
	PAUSE_TIMER(gpuCopyWgtsTimer);
	START_TIMER(gpuShareDeltaTimer);
	PAUSE_TIMER(gpuShareDeltaTimer);
	START_TIMER(gpuLearnTimer);
	PAUSE_TIMER(gpuLearnTimer);
	START_TIMER(gpuReduceWonLossTimer);
	PAUSE_TIMER(gpuReduceWonLossTimer);
	START_TIMER(gpuCompeteTimer);
	PAUSE_TIMER(gpuCompeteTimer);
	START_TIMER(gpuStandingsTimer);
	PAUSE_TIMER(gpuStandingsTimer);

	// update the opponent array
	update_opgrid(g_p.op_method, NULL, 0);
	
	timing_feedback_header(g_p.standings_freq);

	// loop for each learning session
	for (int iSession = 0; iSession < g_p.num_sessions; iSession++) {

//		printf("\n>>>>>>>>>>>>>>> begin session %d <<<<<<<<<<<<<<<<<<\n", iSession);
		timing_feedback_dot(iSession % g_p.standings_freq);
		
		dim3 learnBlockDim(g_p.board_size);
		dim3 learnGridDim(g_p.num_agents, g_p.num_opponents);
		
		dim3 competeBlockDim(g_p.board_size);
		dim3 competeGridDim(g_p.num_agents, g_p.benchmark_ops);

		// learn against other agents simultaneously for segs_per_session iterations
		for (int iSeg = 0; iSeg < g_p.segs_per_session; iSeg++) {

			// copy current agent weights to the saved_wgts area
			if (0 == ((1+iSession) % g_p.refresh_op_wgts_freq)) {
//				printf("\n>>>>>>>>>>>>>>> copy weights\n");
				RESUME_TIMER(gpuCopyWgtsTimer);
				PRE_KERNEL2("copy_wgts_kernel", blockDim_wgtCopy, gridDim_wgtCopy);
				copy_wgts_kernel<<<gridDim_wgtCopy, blockDim_wgtCopy>>>(agGPU->wgts, agGPU->saved_wgts);
				POST_KERNEL(copy_wgts_kernel);		
				
				cudaThreadSynchronize();
				PAUSE_TIMER(gpuCopyWgtsTimer);
			}

//			printf("\n>>>>>>>>>>>>>>> learn for segment %d\n", iSeg);
		
			RESUME_TIMER(gpuLearnTimer);
			PRE_KERNEL2("old_learn_kernel", learnBlockDim, learnGridDim);
			old_learn_kernel<<<learnGridDim, learnBlockDim, dynamic_shared_mem()>>>(agGPU->seeds, agGPU->wgts, agGPU->e, agGPU->saved_wgts, agGPU->wl, agGPU->delta_wgts, g_p.d_opgrid + iSeg * g_p.num_opponents, (0 == iSeg && 0 == iSession % g_p.standings_freq), agGPU->training_turns);
			POST_KERNEL(old_learn_kernel);
			
			cudaThreadSynchronize();
			PAUSE_TIMER(gpuLearnTimer);
			RESUME_TIMER(gpuShareDeltaTimer);


			// reduce the delta_wgts and update agent's weights

//			printf("\n>>>>>>>>>>>>>>> share_delta for segment %d\n", iSeg);
	//		dump_agentsGPU("agents prior to sharing deltas", agGPU, 1, 0);		
			PRE_KERNEL2("share_delta_kernel", blockDim_share, gridDim_share);
			share_delta_kernel<<<gridDim_share, blockDim_share, g_p.num_wgts * sizeof(unsigned)>>>(agGPU->saved_wgts, agGPU->delta_wgts, agGPU->wgts);
			POST_KERNEL(share_delta_kernel);
	//		dump_agentsGPU("agents after sharing deltas", agGPU, 1, 0);
	//
			cudaThreadSynchronize();
			PAUSE_TIMER(gpuShareDeltaTimer);
		}

		// reduce the won-loss results and store in rGPU
		if (0 == (1 + iSession) % g_p.standings_freq) {
//			printf("\n>>>>>>>>>>>>>>> reduce_wl\n");
			RESUME_TIMER(gpuReduceWonLossTimer);
			dim3 wlBlockDim(g_p.num_opponents);
			dim3 wlGridDim(g_p.num_agents);
			PRE_KERNEL2("reduce_wl_kernel", wlBlockDim, wlGridDim);
			reduce_wl_kernel<<<wlGridDim, wlBlockDim, g_p.num_opponents * 3 * sizeof(unsigned)>>>(agGPU->wl, rGPU->standings + iSession * g_p.num_agents, g_p.num_opponents, g_p.half_opponents);
			POST_KERNEL(reduce_wl_kernel); 
			cudaThreadSynchronize();
			PAUSE_TIMER(gpuReduceWonLossTimer);
		}


		// compete against benchmark opponent, storing the results in vsChamp array of rGPU results structure
		if (g_p.benchmark_games > 0 && 0 == ((1+iSession) % g_p.benchmark_freq)) {
//			printf("\n>>>>>>>>>>>>>>> compete\n");
			RESUME_TIMER(gpuCompeteTimer);
//			device_dumpui("agGPU->wl", (unsigned *)agGPU->wl, g_p.num_agents * g_p.benchmark_ops, 4);
			PRE_KERNEL2("compete_kernel", competeBlockDim, competeGridDim);
			compete_kernel<<<competeGridDim, competeBlockDim, dynamic_shared_mem()>>>(agGPU->seeds, agGPU->wgts, d_champ_wgts, agGPU->wl, 0);
			POST_KERNEL(compete_kernel);

//			printf("\n>>>>>>>>>>>>>>> reduce_wl\n");
			dim3 wlBlockDim(g_p.num_opponents);
			dim3 wlGridDim(g_p.num_agents);
//			printf("reducing the benchmark results, run against %d opponents (half ops is %d)\n", g_p.benchmark_ops, g_p.half_benchmark_ops);
			
//			device_dumpui("agGPU->wl", (unsigned *)agGPU->wl, g_p.num_agents * g_p.benchmark_ops, 4);
			
			dim3 wlBlockDim2(g_p.benchmark_ops);
			dim3 wlGridDim2(g_p.num_agents);
			PRE_KERNEL2("reduce_wl_kernel", wlBlockDim2, wlGridDim2);
			reduce_wl_kernel<<<wlGridDim2, wlBlockDim2, g_p.benchmark_ops * 3 * sizeof(unsigned)>>>(agGPU->wl, rGPU->vsChamp + iSession * g_p.num_agents, g_p.benchmark_ops, g_p.half_benchmark_ops);
			POST_KERNEL(reduce_wl_kernel);

//			device_dumpui("total vsChamp results", (unsigned *)(rGPU->vsChamp + iSession * g_p.num_agents), g_p.num_agents, 4);

			cudaThreadSynchronize();
			PAUSE_TIMER(gpuCompeteTimer);

		}

		RESUME_TIMER(gpuStandingsTimer);
		if (0 == ((1+iSession) % g_p.standings_freq)) {

//			printf("\n>>>>>>>>>>>>>>> print_standings\n");
			// copy the standings back to host memory and print them out (which causes the standings to be sorted)
			CUDA_SAFE_CALL(cudaMemcpy(lastStandings, rGPU->standings + iSession * g_p.num_agents, g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(lastVsChamp, rGPU->vsChamp + iSession * g_p.num_agents, g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
			printf("\n\n********** Session %d **********\n", iSession);
//			dump_agent_paramsGPU("agent parameters", agGPU);
			print_standings(lastStandings, lastVsChamp);
			timing_feedback_header(g_p.standings_freq);
		}
		
		// update the opponents used in learning
		if ((g_p.op_method == OM_BEST1 || g_p.op_method == OM_BEST2) && (1+iSession) >= g_p.begin_using_best_ops && 0 == ((1+iSession) % g_p.determine_best_op_freq)){
			printf("\n>>>>>>>>>>>>>>> update opponent grid\n");
			// put the best opponents in g_p.opgrid then copy to the device
			update_opgrid(g_p.op_method, lastStandings, 0);
		}
		cudaThreadSynchronize();
		PAUSE_TIMER(gpuStandingsTimer);
	}

	STOP_TIMER(gpuCopyWgtsTimer, "GPU copy wgts");
	STOP_TIMER(gpuLearnTimer, "GPU learning");
	STOP_TIMER(gpuShareDeltaTimer, "GPU share delta");
	STOP_TIMER(gpuReduceWonLossTimer, "GPU reduce won loss");
	STOP_TIMER(gpuCompeteTimer, "GPU compete vs benchmark");
	STOP_TIMER(gpuStandingsTimer, "GPU standings");
	
#ifdef DUMP_FINAL_AGENTS_GPU
	dump_agentsGPU("agents after learning on GPU", agGPU, 1, DUMP_SAVED_WGTS);
#endif

	
	cudaFree(d_champ_wgts);
	
	return rGPU;
}




