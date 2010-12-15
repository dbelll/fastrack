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

//static unsigned g_seeds[4] = {0, 0, 0, 0};
static PARAMS g_p;

// allowable moves expressed as delta from initial position
// maximum distance left/right is board_width/2 - 1
// maximum distance forward/backward is board_width/2 - 1
// {0, 0} is an allowable move, meaning no piece is moved (ie 'pass')
//static int g_allowable_moves[8][2] = {{-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}};
static int g_allowable_moves[8][2] = MOVES_KNIGHT;
static unsigned *g_moves = NULL;
//static unsigned *g_moves_op = NULL; // used if there is directional difference to possible moves

static unsigned g_bits_for_board_ints[2];	// stores the number of valid bits in each int that makes up a board

#pragma mark -
#pragma mark misc.

/*
	Initialize the global seeds using specified seed value.
*/
//void set_global_seeds(unsigned seed)
//{
//	srand(seed);
//	for (int i = 0; i < 4; i++) {
//		g_seeds[i] = rand();
//	}
//}

float sigmoid(float x)
{
	return 1.0f/(1.0f + expf(-x));
}

#pragma mark -
#pragma mark allocating and freeing

void freeAgentGPU(AGENT *ag)
{
	if (ag) {
		if (ag->seeds) CUDA_SAFE_CALL(cudaFree(ag->seeds));
		if (ag->wgts) CUDA_SAFE_CALL(cudaFree(ag->wgts));
		if (ag->e) CUDA_SAFE_CALL(cudaFree(ag->e));
		if (ag->alpha) CUDA_SAFE_CALL(cudaFree(ag->alpha));
		if (ag->epsilon) CUDA_SAFE_CALL(cudaFree(ag->epsilon));
		if (ag->lambda) CUDA_SAFE_CALL(cudaFree(ag->lambda));
		CUDA_SAFE_CALL(cudaFree(ag));
	}
}

void freeAgentCPU(AGENT *ag)
{
	if (ag) {
		if (ag->state) free(ag->state);
		if (ag->seeds) free(ag->seeds);
		if (ag->wgts) free(ag->wgts);
		if (ag->e) free(ag->e);
		if (ag->alpha) free(ag->alpha);
		if (ag->epsilon) free(ag->epsilon);
		if (ag->lambda) free(ag->lambda);
		free(ag);
	}
}

void freeCompactAgent(COMPACT_AGENT *ag)
{
	if (ag) {
		if (ag->seeds) free(ag->seeds);
		if (ag->fdata) free(ag->fdata);
		free(ag);
	}
}


#pragma mark -
#pragma mark game functions

// Calcualte the value for a state s using the specified weights,
// storing hidden activation in the specified location and returning the output value
float val_for_state(float *wgts, unsigned *state, float *hidden)
{
//	printf("calculating value for state...\n");
	float out = 0.0f;
	
	for (unsigned iHidden = 0; iHidden < g_p.num_hidden; iHidden++) {
		// first add in the bias
		hidden[iHidden] = -1.0f * wgts[iHidden];
		
		// next, loop over all bits in the state, and add in their contribution
		unsigned iWgt = g_p.num_hidden;	// index into wgts for idx = 0
		for (int i = 0; i < g_p.state_size; i++) {
			unsigned bitMax = (0 == (i % g_p.board_ints)) ? ((g_p.board_size > 32) ? 32 : g_p.board_size)
															: (g_p.board_size - 32);
			unsigned s = state[i];
//			printf("state[%d] is %u\n", i, s);
			unsigned iBit = 0;
			while (iBit < bitMax) {
				if (s & 1) hidden[iHidden] += wgts[iHidden + iWgt];
//				if (s & 1) printf("bit %d of byte %d is on!  adding %9.4f to hidden node %d (now = %9.4f)\n", iBit, i, wgts[iHidden + iWgt], iHidden, hidden[iHidden]);
				s >>= 1;
				iBit++;
				iWgt += g_p.num_hidden;
			}
		}
//		printf("input to hidden[%d] = %9.4f", iHidden, hidden[iHidden]);
		// next, apply the activation function
		hidden[iHidden] = sigmoid(hidden[iHidden]);
//		printf("after sigmoid applied its %9.4f\n", hidden[iHidden]);
		// now add this hidden node's contribution to the output
		out += hidden[iHidden] * wgts[iHidden + iWgt];
	}
	
	// finally, add the bias to the output value and apply the sigmoid function
	out += -1.0f * wgts[g_p.alloc_wgts - g_p.num_hidden];
//	printf("input to output node is %9.4f, %9.4f after sigmoid\n", out, sigmoid(out));
	return sigmoid(out);
}

unsigned int_for_cell(unsigned row, unsigned col)
{
	return (col + row * g_p.board_width) / 32;
}

unsigned bit_for_cell(unsigned row, unsigned col)
{
	return (col + row * g_p.board_width) % 32;
}

unsigned val_for_cell(unsigned row, unsigned col, unsigned *board)
{
	return board[int_for_cell(row, col)] & (1 << (bit_for_cell(row, col)));
}

void set_val_for_cell(unsigned row, unsigned col, unsigned *board, unsigned val)
{
	if (val) {
		board[int_for_cell(row, col)] |= (1 << (bit_for_cell(row, col)));
	}else {
		board[int_for_cell(row, col)] &= ~(1 << (bit_for_cell(row, col)));
	}
}

char char_for_cell(unsigned row, unsigned col, unsigned *state)
{
	unsigned s0 = val_for_cell(row, col, state);
	unsigned s1 = val_for_cell(row, col, state + g_p.board_ints);
	if (s0 && s1) return '?';
	else if (s0) return 'X';
	else if (s1) return 'O';
	return '.';
}

// add n pieces to un-occupied cells of a board
void random_add(unsigned *board, unsigned n)
{
//	printf("random_add %d pieces\n", n);
	while (n > 0) {
//		printf("%d pieces left to add to board\n", n);
		unsigned i = rand() % g_p.board_bits;
//		printf("   i is %d\n", i);
		unsigned iBitMask = 1 << (i & 31);	// mod 32
//		printf("   iBitMask is %u\n", iBitMask);
		unsigned iInt = i >> 5;	// divided by 32
//		printf("   iInt is %d\n", iInt);
		if (!(board[iInt] & iBitMask)) {
			board[iInt] |= iBitMask;
			--n;
		}
//		dump_board(board);
	}
}

// generate a random board
void random_board(unsigned *board, unsigned n)
{
	printf("random_board for %d pieces\n", n);
	// first, empty the board
	for (int i = 0; i < g_p.board_ints; i++) board[i] = 0;
	
	// now add a random, non-occupied cell
	random_add(board, n);
}


// generate a random board, avoiding any occupied cells in the mask
void random_board_masked(unsigned *board, unsigned *mask, unsigned n)
{
	// first, copy the mask to the board
	for (int i = 0; i < g_p.board_ints; i++) board[i] = mask[i];

	// now add a random, non-occupied cell
	random_add(board, n);
	
	// XOR away the mask
	for (int i = 0; i < g_p.board_ints; i++) board[i] ^= mask[i];
}

// generate a random state with n pieces for ech player
void random_state(unsigned *state, unsigned n)
{
	random_board(state, n);
	random_board_masked(state + g_p.board_ints, state, n);
}

// return a read-only mask for the specified number of cols on the right
unsigned *mask_cols_right(unsigned cols)
{
	static unsigned need_init = 1;
	static unsigned *mask;
	if (need_init) {
		mask = (unsigned *)calloc(g_p.board_ints * (g_p.board_width-1), sizeof(unsigned));
		for (int delta = 1; delta < g_p.board_width; delta++) {
			for (int col = g_p.board_width - delta; col < g_p.board_width; col++) {
				for (int row = 0; row < g_p.board_height; row++) {
					set_val_for_cell(row, col, mask + (delta-1)*g_p.board_ints, 1);
				}
			}
		}
	}
	return mask + (cols-1)*g_p.board_ints;
}

// return a read-only mask for the specified number of cols on the left
unsigned *mask_cols_left(unsigned cols)
{
	static unsigned need_init = 1;
	static unsigned *mask;
	if (need_init) {
		mask = (unsigned *)calloc(g_p.board_ints * (g_p.board_width-1), sizeof(unsigned));
		for (int delta = 1; delta < g_p.board_width; delta++) {
			for (int col = 0; col < delta; col++) {
				for (int row = 0; row < g_p.board_height; row++) {
					set_val_for_cell(row, col, mask + (delta-1)*g_p.board_ints, 1);
				}
			}
		}
	}
	return mask + (cols-1)*g_p.board_ints;
}

// a mask for the unused bits above the board
unsigned *mask_rows_top()
{
	static unsigned *mask;
	if (!mask) {
		mask = (unsigned *)calloc(g_p.board_ints, sizeof(unsigned));
		if (g_p.board_unused > 0) {
			mask[g_p.board_ints-1] = - (1 << (32 - g_p.board_unused));
		}
	}
	return mask;
}

// shift the board by a number of columns
void colshift(unsigned *board, int delta)
{
//	printf("colshift -- initial board:\n");
//	dump_board(board);
	
	if (delta < 0) {
		delta = -delta;

//		printf("shifting the board %d columns to the left\n", delta);

		// shift the low int to the left (lo-bits) first
		board[0] >>= delta;
		
//		printf("board after shifting the low int to the left\n");
//		dump_board(board);
		
		// get a mask to make the new columns blank
		unsigned *mask = mask_cols_right(delta);

		if (g_p.board_ints > 1) {
			// get the bits that will shift out of the hi int
			unsigned bits = ((1 << delta) - 1) & board[1];
			// align them to the highest bits
			bits <<= (32 - delta);
			// add them to the lo int
			board[0] |= bits;
			// now shift the hi int by delta
			board[1] >>= delta;
			// mask off the bits in the new cols
			board[1] &= ~mask[1];
		}
		board[0] &= ~mask[0];
	}else if (delta > 0) {
	
//		printf("shifting the board %d columns to the right\n", delta);

		// shift the high int to the right (hi-bits) first
		unsigned *col_mask = mask_cols_left(delta);

//		printf("col_mask:\n");
//		dump_board(col_mask);

		unsigned *row_mask = mask_rows_top();

		if (g_p.board_ints > 1) {
			board[1] <<= delta;
		
//			printf("board after shifting the hi int to the right\n");
//			dump_board(board);
		
			// get the bits that will shift out of the lo int
			unsigned bits = -(1 << (32-delta)) & board[0];
			// align them to the lo bits
			bits >>= 32-delta;
			// add them to the hi int
			board[1] |= bits;
			
//			printf("board after adding the bits that shifted from lo int to hi int\n");
//			dump_board(board);
			
			// mask off the rows above the board and cols on the left
			board[1] &= ((~row_mask[1]) & (~col_mask[1]));
			
//			printf("board after masking off the hi int\n");
//			dump_board(board);
		}
		board[0] <<= delta;

//		printf("board after shifting the lo int to the right\n");
//		dump_board(board);

		board[0] &= ~row_mask[0] & ~col_mask[0];

//		printf("board after masking off the lo int\n");
//		dump_board(board);
	}
}

void rowshift(unsigned *board, int delta)
{
	colshift(board, delta * g_p.board_width);
}

void shift_board(unsigned *board, int colDelta, int rowDelta)
{
	colshift(board, colDelta + g_p.board_width * rowDelta);
}

void shift_state(unsigned *state, int colDelta, int rowDelta)
{
	shift_board(state, colDelta, rowDelta);
	shift_board(state + g_p.board_ints, colDelta, rowDelta);
}


// return a pointer to the starting state for the game
unsigned *start_state()
{
	static unsigned *ss = NULL;
	if (ss == NULL) {
		ss = (unsigned *)calloc(g_p.state_size, sizeof(unsigned));
		for (int col = 0; col < g_p.board_width; col++) {
			set_val_for_cell(0, col, ss, 1);
			set_val_for_cell(g_p.board_height-1, col, ss + g_p.board_ints, 1); 
		}
	}
	return ss;
}
// copy the starting state to the provided location
void copy_start_state(unsigned *state)
{
	bcopy(start_state(), state, g_p.state_size * sizeof(unsigned));
}

void switch_sides(unsigned *state)
{
	for (int i = 0; i < g_p.board_ints; i++) {
		unsigned temp = state[i];
		state[i] = state[i + g_p.board_ints];
		state[i + g_p.board_ints] = temp;
	}
}
void copy_state(unsigned *to, unsigned *from)
{
	for (int i = 0; i < g_p.state_size; i++) {
		to[i] = from[i];
	}
}

unsigned is_empty(unsigned *board)
{
	for (int i = 0; i < g_p.board_ints; i++) {
		if (board[i]) return 0;
	}
	return 1;
}

unsigned not_empty(unsigned *board)
{
	if (board[0]) return 1;
	if (g_p.board_ints > 1 && board[1]) return 1;
	return 0;
}

unsigned *empty_board()
{
	static unsigned *e = NULL;
	if (!e) {
		e = (unsigned *)calloc(g_p.board_ints, sizeof(unsigned));
	}
	return e;
}

// Calculate the reward for the given state (from X's perspective)
// non-zero reward ==> terminal state
int reward(unsigned *state)
{
//	printf("calculate reward for :\n");
//	dump_state(state);
	int reward = 0;
	if (is_empty(state)) reward = 100;
	if (is_empty(state + g_p.board_ints)) reward = -100;
//	printf("reward is %d\n", reward);
	return reward;
}

#pragma mark -
#pragma mark dump stuff

void dump_col_header(unsigned leftMargin, unsigned nCols)
{
	while (leftMargin-- > 0) {
		printf(" ");
	}
	for (int i = 0; i < nCols; i++) {
		printf(" %c", 'a' + i);
	}
	printf("\n");
}

void dump_boards(unsigned *b1, unsigned *b2)
{
	unsigned *state = (unsigned *)malloc(2 * g_p.state_size * sizeof(unsigned));
	bcopy(b1, state, g_p.state_size * sizeof(unsigned));
	bcopy(b2, state + g_p.state_size, g_p.state_size * sizeof(unsigned));
	dump_state(state);
}

void dump_state_ints(unsigned *state)
{
	printf("[STATE");
	for (int i = 0; i < g_p.state_size; i++) {
		printf("%11u", state[i]);
	}
	printf("\n");
}

void dump_state(unsigned *state)
{
//	printf("dump_state for %u %u %u %u\n", state[3], state[2], state[1], state[0]);
	dump_col_header(3, g_p.board_width);
	for (int row = g_p.board_height - 1; row >= 0; row--) {
		printf("%2u ", row+1);
		for (int col = 0; col < g_p.board_width; col++) {
			printf(" %c", char_for_cell(row, col, state));
		}
		printf("%3u", row+1);
		printf("\n");
	}
	dump_col_header(3, g_p.board_width);
//	printf("[STATE");
//	for (int i = 0; i < g_p.state_size; i++) {
//		printf("%11u", state[i]);
//	}
//	printf("\n");
	dump_state_ints(state);
}

void dump_board(unsigned *board)
{
	dump_col_header(3, g_p.board_width);
	for (int row = g_p.board_height - 1; row >= 0; row--) {
		printf("%2u ", row+1);
		for (int col = 0; col < g_p.board_width; col++) {
			printf(" %c", val_for_cell(row, col, board) ? 'X' : '.');
		}
		printf("%3u", row+1);
		printf("\n");
	}
	dump_col_header(3, g_p.board_width);
}


void dump_wgts_header(const char *str)
{
	printf("%s", str);
	for (int i = 0; i < g_p.num_hidden; i++) {
		printf(",  %6d  ", i);
	}
	printf("\n");
}

void dump_wgts(float *wgts)
{
	for (int i = 0; i < g_p.num_hidden; i++) {
		printf(", %9.4f", wgts[i]);
	}
	printf("\n");
}

void dump_agent(AGENT *agCPU, unsigned iag, unsigned dumpW)
{
	printf("[SEEDS], %10d, %10d %10d %10d\n", agCPU->seeds[iag], agCPU->seeds[iag + g_p.num_agents], agCPU->seeds[iag + 2 * g_p.num_agents], agCPU->seeds[iag + 3 * g_p.num_agents]);
	printf("[STATE]");

	dump_wgts_header("[ WEIGHTS]");
	// get the weight pointer for this agent
	float *pWgts = agCPU->wgts + iag * g_p.alloc_wgts;
	printf("[    B->H]"); dump_wgts(pWgts);
	for (int i = 0; i < g_p.num_inputs; i++){
		printf("[IN%03d->H]", i); dump_wgts(pWgts + (1+i) * g_p.num_hidden);
	}
	printf("[    H->O]"); dump_wgts(pWgts + (1+g_p.num_inputs) * g_p.num_hidden);
	printf("[    B->O], %9.4f\n\n", pWgts[(2+g_p.num_inputs) * g_p.num_hidden]);

	if (dumpW) {
		dump_wgts_header("[    W    ]");
		// get the W pointer for this agent
		float *pW = agCPU->e + iag * g_p.alloc_wgts;
		printf("[    B->H]"); dump_wgts(pW);
		for (int i = 0; i < g_p.num_inputs; i++){
			printf("[IN%03d->H]", i); dump_wgts(pW + (1+i) * g_p.num_hidden);
		}
		printf("[    H->O]"); dump_wgts(pW + (1+g_p.num_inputs) * g_p.num_hidden);
		printf("[    B->O], %9.4f\n\n", pW[(2+g_p.num_inputs) * g_p.num_hidden]);
	}

	printf("[   alpha], %9.4f\n", agCPU->alpha[iag]);
	printf("[ epsilon], %9.4f\n", agCPU->epsilon[iag]);
	printf("[  lambda], %9.4f\n\n", agCPU->lambda[iag]);
}

void dump_agentsCPU(const char *str, AGENT *agCPU, unsigned dumpW)
{
	printf("======================================================================\n");
	printf("%s\n", str);
	printf("----------------------------------------------------------------------\n");
	for (int i = 0; i < g_p.num_agents; i++) {
		printf("[AGENT%5d]\n", i);
		dump_agent(agCPU, i, dumpW);
	}
	printf("======================================================================\n");
	
}

void dump_compact_agent(COMPACT_AGENT *ag)
{
	printf("[SEEDS], %10d, %10d %10d %10d\n", ag->seeds[0], ag->seeds[1], ag->seeds[2], ag->seeds[3]);
	printf("[    B->H]"); dump_wgts(ag->fdata + ag->iWgts);
	for (int i = 0; i < g_p.num_inputs; i++){
		printf("[IN%03d->H]", i); dump_wgts(ag->fdata +ag->iWgts + (1+i) * g_p.num_hidden);
	}
	printf("[    H->O]"); dump_wgts(ag->fdata + ag->iWgts + (1+g_p.num_inputs) * g_p.num_hidden);
	printf("[    B->O], %9.4f\n", ag->fdata[ag->iWgts + (2+g_p.num_inputs) * g_p.num_hidden]);
	printf("[   alpha], %9.4f\n", ag->fdata[ag->iAlpha]);
	printf("[ epsilon], %9.4f\n", ag->fdata[ag->iEpsilon]);
	printf("[  lambda], %9.4f\n", ag->fdata[ag->iLambda]);
}


void dumpResults(RESULTS *row)
{
	printf("Best agents each round...\n");
	for (int i = 0; i < row->allocated; i++) {
		if (row->best+i){
			printf("[ROUND%3d]\n", i);
			dump_compact_agent(row->best+i);
		}
	}
}


#pragma mark -
#pragma mark CPU - setup
RESULTS *newResults()
{
	RESULTS *row = (RESULTS *)malloc(sizeof(RESULTS));
	row->allocated = g_p.num_episodes;
	row->best = (COMPACT_AGENT *)malloc(row->allocated * sizeof(COMPACT_AGENT));
	return row;
}

void freeResults(RESULTS *row)
{
	if (row) {
		for (int i = 0; i < row->allocated; i++)
			if (row->best + i) freeCompactAgent(row->best+i);
	
		free(row);
	}
}

// set the global values for number of valid bits in each board int
void calc_bits_for_board_ints()
{
	g_bits_for_board_ints[0] = (g_p.board_size > 32) ? 32 : g_p.board_size;
	if (g_p.board_ints == 1) {
		g_bits_for_board_ints[1] = 0;
	}else {
		g_bits_for_board_ints[1] = g_p.board_size - 32;
	}
}

// calculates agent pointers based on offset from ag->wgts
// agent data is organized as follows
void set_agent_float_pointers(AGENT *ag)
{
	ag->e = ag->wgts + g_p.alloc_wgts * g_p.num_agents;
	ag->alpha = ag->e + g_p.alloc_wgts * g_p.num_agents;
	ag->epsilon = ag->alpha + g_p.num_agents;
	ag->lambda = ag->epsilon + g_p.num_agents;
}

/*
	Build the move array, g_moves, using the allowable moves in g_allowable_moves[8][2]
	g_moves will be of size board_size * 8 * board_ints
*/
void build_move_array()
{
//	char buf1[3];
//	buf1[2] = 0;
//	char buf2[3];
//	buf2[2] = 0;
	g_moves = (unsigned *)calloc(g_p.board_size * 8 * g_p.board_ints, sizeof(unsigned));
	for (int row = 0; row < g_p.board_height; row++) {
		for (int col = 0; col < g_p.board_width; col++) {
			for (int m = 0; m < 8; m++) {
				unsigned *board = g_moves + g_p.board_ints * (m + 8*(col + g_p.board_width * row));
				int toCol = col + g_allowable_moves[m][0];
				int toRow = row + g_allowable_moves[m][1];
				
				if (toCol >= 0 && toCol < g_p.board_width && toRow >= 0 && toRow < g_p.board_height) {
//					printf("valid move from %s to %s\n", move_string(buf1, col, row), move_string(buf2, toCol, toRow));
//					set_val_for_cell(row, col, board, 1);
					set_val_for_cell(toRow, toCol, board, 1);
//					dump_board(board);
				}
			}
		}
	}
}
AGENT *init_agentsCPU(PARAMS p)
{
	printf("init_agentsCPU...\n");
	// save the parameters and calculate any global constants based on the parameters
	g_p = p;
	calc_bits_for_board_ints();
	build_move_array();

	// allocate and initialize the agent data on CPU
	AGENT *ag = (AGENT *)malloc(sizeof(AGENT));
	unsigned count = 4*p.num_agents;
	ag->seeds = (unsigned *)malloc(count * sizeof(unsigned));
	for (int i = 0; i < 4*p.num_agents; i++) ag->seeds[i] = rand();
	
	ag->state = (unsigned *)malloc(p.num_agents * p.state_size * sizeof(unsigned));
	
//	printf("seeds allocated at %p\n", ag->seeds);
	
	// allocate one chunk of float data and set up pointers to appropriate parts of that chunk
	count = (2*p.alloc_wgts + 3) * p.num_agents;
	ag->wgts = (float *)malloc(count * sizeof(float));
	set_agent_float_pointers(ag);
	
//	printf("total of %d float values allocated\n", count);
//	printf("float values allocated at %p, first value is %f\n", ag->wgts, ag->wgts[0]);
//	printf("other pointers ag->e is %p\n", ag->e);
//	printf("           ag->alpha is %p\n", ag->alpha);
//	printf("         ag->epsilon is %p\n", ag->epsilon);
//	printf("          ag->lambda is %p\n", ag->lambda);
	
	// initialize values
	printf("initializing weights for %d values with min of %f and max of %f\n", p.alloc_wgts * p.num_agents, p.init_wgt_min, p.init_wgt_max);
	for (int i=0; i < p.alloc_wgts * p.num_agents; i++){
//		printf("%d ", i);
		ag->wgts[i] = rand_wgt2(p.init_wgt_min, p.init_wgt_max);
//		printf("- ");
		ag->e[i] = 0.0f;
	}
	
	printf("weights and W have been initialized\n");
	
	for (int i = 0; i < p.num_agents; i++) {
		ag->alpha[i] = p.alpha;
		ag->epsilon[i] = p.epsilon;
		ag->lambda[i] = p.lambda;
	}
	
	printf("alpha, epsilon, and lambda have been initialized\n");
	
	return ag;
}

char *move_string(char *buff, unsigned col, unsigned row)
{
	buff[0] = 'a' + col;
	buff[1] = '1' + row;
	return buff;
}


#pragma mark CPU - run

/*
	calculate the values for every possible next state, and put the best one into nextState, returning its value
*/
float choose_move(unsigned *currState, float *wgts)
{
	static unsigned *to = NULL;
	static unsigned *temp = NULL;
	static float *hidden_activation = NULL;
	static unsigned *bestMove = NULL;
	
	unsigned nextState[MAX_STATE_SIZE];
	
	if (!to) to = (unsigned *)calloc(g_p.board_ints * 8, sizeof(unsigned));
	if (!temp) temp = (unsigned *)calloc(g_p.board_ints * 8, sizeof(unsigned));
	if (!hidden_activation) hidden_activation = (float *)malloc(g_p.num_hidden * sizeof(float));
	if (!bestMove) bestMove = (unsigned *)calloc(g_p.state_size, sizeof(unsigned));
	
	unsigned *boardA = currState;
	unsigned *boardB = currState + g_p.board_ints;
	unsigned *nextA = nextState;
	unsigned *nextB = nextState + g_p.board_ints;
	
	unsigned noVal = 1;
	float bestVal = 0.0f;
	unsigned from[2];
	from[0] = 0; from[1] = 0;
	unsigned move_count = 0;
	
	// loop through all the possible piece positions
	for (int iInt = 0; iInt < g_p.board_ints; iInt++) {
		for (int iBit = 0; iBit < g_bits_for_board_ints[iInt]; iBit++) {
			unsigned iPo = iBit + iInt * 32;
//			printf("iByte is %d, iBit is %d, iPo = %d\n", iByte, iBit, iPo);
			from[iInt] = 1 << iBit;
//			printf("from is");
//			for (int i = 0; i < g_p.board_ints; i++) {
//				printf(" %u", from[i]);
//			}
//			printf("\n");
			
//			printf("move from ...\n");
//			dump_board(from);
//			printf("???\n");
			
			if (from[iInt] & boardA[iInt]) {
//				printf("\n\n\ncan move from this spot:\n");
//				dump_board(from);
				
				// there is a piece at this position
				for (int i = 0; i < 8; i++) {
					for (int j = 0; j < g_p.board_ints; j++) {
						unsigned iTo = j + g_p.board_ints * i;
						// copy the possible moves into the local memory
						to[iTo] = g_moves[iPo * 8 * g_p.board_ints + iTo];
						// see if move destination is available
						temp[iTo] = to[iTo] & ~boardA[j];
						nextA[j] = (boardA[j] & ~from[j]) | to[iTo];
						nextB[j] = boardB[j] & ~to[iTo];
					}
//					printf("temp for possible move %d\n", i);
//					dump_board(temp + i * g_p.board_ints);
					if (not_empty(temp + i * g_p.board_ints)) {
						
//						printf("boardA:\n");
//						dump_board(boardA);
//						
//						printf("from:\n");
//						dump_board(from);
//						
//						printf("to:\n");
//						dump_board(to + i * g_p.board_ints);
//						
//						printf("results in nextA:\n");
//						dump_board(nextA);
//						
//						printf("possible new board:\n");
//						dump_state(nextState);
						++move_count;
						
						float val = val_for_state(wgts, nextState, hidden_activation);
						if (noVal || val > bestVal) {
							noVal = 0;
							bestVal = val;
							for (int k = 0; k < g_p.state_size; k++) {
								bestMove[k] = nextState[k];
							}
							if (g_p.board_ints > 1) bestMove[1] = nextState[1];
						}
//						printf("value is %f\n", val_for_state(wgts, nextState, hidden_activation));
					}
				}
			}
		}
		from[iInt] = 0;
	}
//	printf("\n====================\n%d possible moves\n", move_count);
//	printf("best move with value %f is:\n", bestVal);
//	dump_state(bestMove);
	for (int k = 0; k < g_p.state_size; k++) {
		currState[k] = bestMove[k];
	}
//	printf("====================\n");
	return bestVal;
}

// take an action from the specified state, returning the reward
float take_action(unsigned *state, float *wgts)
{
	float r = reward(state);
	if (r) return r;	// given state is terminal, just return the reward
	switch_sides(state);
	choose_move(state, wgts);
	switch_sides(state);
	return -reward(state);
}

void set_start_state(unsigned *state, unsigned pieces)
{
	(pieces == 0) ? copy_start_state(state) : random_state(state, pieces);
}


// update eligibility traces using the activation values for hidden and output nodes
void update_trace(unsigned state128float *wgts, float *e, float *hidden, float *output)
{
	// first decay all existing values
	
	// next update the weights from hidden layer to output node
	// first the bias
	float g_prime_i = output[0] * (1.0f - output[0]);
	unsigned iH2O = (2 * g_p.board_size + 1) * g_p.num_hidden;
	e[iH2O + g_p.num_hidden] += -1.0f * g_prime_i;
	
	// next do all the hidden nodes to output node
	for (int j = 0; j < g_p.num_hidden; j++) {
		e[iH2O + j] += hidden[j] * g_prime_i;
	}
	
	// now update the weights to the hidden nodes
	for (int j = 0; j < g_p.num_hidden; j++) {
		float error_j = wgts[iH2O + j] * g_prime_i;
		float g_prime_j = hidden[j]*(1.0f - hidden[j]);
		
		// first the bias to the hidden node
		e[j] += -1.0f * g_prime_j;
		
		// then all the input -> hidden values
		for (int k = 0; k < g_p.board_size * 2; k++) {
			<#statements#>
		}
	}
	
}


// run a learning session using agent 0 playing against itself
// Start with a random board with start_pieces per side, or the normal starting board if start_pieces is 0
// Top the learning after num_turns turns (for each player)
void auto_learn(AGENT *agCPU, unsigned start_pieces, unsigned num_turns)
{
	// just one agent is used for this initial coding
	
	// set up the starting state
	set_start_state(agCPU->state, start_pieces);

	printf("starting board:\n");
	dump_state(agCPU->state);		
	
	// choose the action, storing the next state in agCPU->state and returning the value for the next state
	float V = choose_move(agCPU->state, agCPU->wgts);
	
	// update eligibility trace here
	
	// loop over the number of turns
	while (num_turns--) {
		printf("\n\n------- %d turns left -------\n", num_turns+1);
		printf("after own move:\n");
		dump_state(agCPU->state);		

		float reward = take_action(agCPU->state, agCPU->wgts);

		printf("after opponent move:\n");
		dump_state(agCPU->state);

		if (reward){
			printf("\n\n****** GAME OVER with r = %9.4f *******\n\n\n", reward);
			set_start_state(agCPU->state, start_pieces);
			printf("new starting board:\n");
			dump_state(agCPU->state);		
		}
		float V_prime = choose_move(agCPU->state, agCPU->wgts);
		float delta = reward + (reward ? 0.0f : (g_p.gamma * V_prime)) - V;
		// update theta's here
//		if (reward) //reset trace here 
		// update trace here
		V = V_prime;
	}
	printf("\n\n\n");
}


RESULTS *runCPU(AGENT *agCPU)
{
	unsigned state[MAX_STATE_SIZE];
//	state[0] = (unsigned *)malloc(g_p.state_size * sizeof(unsigned));
//	state[1] = (unsigned *)malloc(g_p.state_size * sizeof(unsigned));

//	random_board(state[0], g_p.board_width);
//	random_board_masked(state[0] + g_p.board_ints, state[0], g_p.board_width);
//	random_state(state[0], g_p.board_width);
	copy_state(state, start_state());
	dump_state(state);
	
	// test with agent 0
//	choose_move(state[currState], state[1-currState], agCPU->wgts);
	
	dump_agent(agCPU, 0, 0);

	auto_learn(agCPU, 0, 10);
	auto_learn(agCPU, 1, 10);
	auto_learn(agCPU, 2, 10);
	auto_learn(agCPU, 3, 10);
	
//	unsigned move = 0;
//	while (0 == reward(state) && move < 50) {
//		printf("\n\nturn %d:\n", move++);
//		choose_move(state, agCPU->wgts);
//		printf("after own move:\n");
//		dump_state(state);		
//		switch_sides(state);
////			printf("board after switching sides:\n");
////			dump_state(state[currState]);
//		choose_move(state, agCPU->wgts);
////			printf("after the move....\n");
////			dump_state(state[currState]);
//		switch_sides(state);
////			printf("switching sides back....\n");
////			dump_state(state[currState]);
//		printf("after opponent's move:\n");
//		dump_state(state);		
//	}
	return NULL;
}



#pragma mark -
#pragma mark GPU - Only

AGENT *init_agentsGPU(AGENT *agCPU)
{
	AGENT *agGPU = (AGENT *)malloc(sizeof(AGENT));
	agGPU->seeds = device_copyui(agCPU->seeds, 4 * g_p.num_agents);
	agGPU->wgts = device_copyf(agCPU->wgts, g_p.agent_float_count * g_p.num_agents);
	set_agent_float_pointers(agGPU);
	
	
	return agGPU;
}


RESULTS *runGPU(AGENT *agGPU)
{
	return NULL;
}




