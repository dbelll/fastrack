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

// g_moves will be the move array of size board_size * 8 (for up to 8 possible moves from each cell)
// The values stored is the cell number where the piece would be after the move.  The board_size rows
// in the table are for all the possible starting cell numbers.  A move of -1 is not valid
static int *g_moves = NULL;

static unsigned *g_start_state = NULL;

//static unsigned *g_moves_op = NULL; // used if there is directional difference to possible moves

//static unsigned g_bits_for_board_ints[2];	// stores the number of valid bits in each int that makes up a board

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

unsigned index4rc(unsigned row, unsigned col)
{
	return row * g_p.board_width + col;
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

unsigned is_empty(unsigned *board)
{
	for (int i = 0; i < g_p.board_size; i++) {
		if (board[i]) return 0;
	}
	return 1;
}

unsigned not_empty(unsigned *board){ return !is_empty(board); }

// Calculate the reward for the given state (from X's perspective)
// non-zero reward ==> terminal state
float reward(unsigned *state, unsigned *terminal)
{
	float reward = 0.0f;
	*terminal = 0;
	if (is_empty(O_BOARD(state))){ reward = 1.0f; *terminal = 1; }
	if (is_empty(X_BOARD(state))){ reward = 0.0f; *terminal = 1; }
	return reward;
}

// Calcualte the value for a state s using the specified weights,
// storing hidden activation in the specified location and returning the output value
float val_for_state(float *wgts, unsigned *state, float *hidden, float *out)
{
//	printf("calculating value for state...\n");
	unsigned terminal;
	float r = reward(state, &terminal);
	if (terminal) return r;

	out[0] = 0.0f;
	
	for (unsigned iHidden = 0; iHidden < g_p.num_hidden; iHidden++) {
		// first add in the bias
		hidden[iHidden] = -1.0f * wgts[iHidden];

		// next loop update for all the input nodes
		for (int i = 0; i < g_p.board_size * 2; i++) {
			if (state[i]) {
				hidden[iHidden] += wgts[iHidden + g_p.num_hidden * (1 + i)];
			}
		}
		
		// apply the sigmoid function
		hidden[iHidden] = sigmoid(hidden[iHidden]);

		// accumulate into the output
		out[0] += hidden[iHidden] * wgts[iHidden + g_p.num_hidden * (1 + g_p.board_size * 2)];
	}
	
	// finally, add the bias to the output value and apply the sigmoid function
	out[0] += -1.0f * wgts[g_p.num_wgts - g_p.num_hidden];
	out[0] = sigmoid(out[0]);
	return out[0];
}

//unsigned int_for_cell(unsigned row, unsigned col)
//{
//	return (col + row * g_p.board_width) / 32;
//}
//
//unsigned bit_for_cell(unsigned row, unsigned col)
//{
//	return (col + row * g_p.board_width) % 32;
//}

unsigned val_for_cell(unsigned row, unsigned col, unsigned *board)
{
	return board[index4rc(row, col)];
}

void set_val_for_cell(unsigned row, unsigned col, unsigned *board, unsigned val)
{
	board[index4rc(row, col)] = val;
}

char char_for_index(unsigned i, unsigned *state)
{
	unsigned s0 = X_BOARD(state)[i];
	unsigned s1 = O_BOARD(state)[i];
	if (s0 && s1) return '?';
	else if (s0) return 'X';
	else if (s1) return 'O';
	return '.';
}

char char_for_cell(unsigned row, unsigned col, unsigned *state)
{
	return char_for_index(index4rc(row, col), state);
}

// add n pieces to un-occupied cells of a board
void random_add(unsigned *board, unsigned n)
{
	while (n > 0) {
		unsigned i = rand() % g_p.board_size;
		if (!board[i]) {
			board[i] = 1;
			--n;
		}
	}
}

// generate a random board
void random_board(unsigned *board, unsigned n)
{
	// first, empty the board
	for (int i = 0; i < g_p.board_size; i++) board[i] = 0;
	
	// now add a random, non-occupied cell
	random_add(board, n);
}


// generate a random board, avoiding any occupied cells in the mask
void random_board_masked(unsigned *board, unsigned *mask, unsigned n)
{
	// first, copy the mask to the board
	for (int i = 0; i < g_p.board_size; i++) board[i] = mask[i];

	// now add a random, non-occupied cell
	random_add(board, n);
	
	// XOR away the mask
	for (int i = 0; i < g_p.board_size; i++) board[i] ^= mask[i];
}

// generate a random state with n pieces for ech player
void random_state(unsigned *state, unsigned n)
{
	random_board(X_BOARD(state), n);
	random_board_masked(O_BOARD(state), X_BOARD(state), n);
}

// return a read-only mask for the specified number of cols on the right
//unsigned *mask_cols_right(unsigned cols)
//{
//	static unsigned need_init = 1;
//	static unsigned *mask;
//	if (need_init) {
//		mask = (unsigned *)calloc(g_p.board_ints * (g_p.board_width-1), sizeof(unsigned));
//		for (int delta = 1; delta < g_p.board_width; delta++) {
//			for (int col = g_p.board_width - delta; col < g_p.board_width; col++) {
//				for (int row = 0; row < g_p.board_height; row++) {
//					set_val_for_cell(row, col, mask + (delta-1)*g_p.board_ints, 1);
//				}
//			}
//		}
//	}
//	return mask + (cols-1)*g_p.board_ints;
//}
//
// return a read-only mask for the specified number of cols on the left
//unsigned *mask_cols_left(unsigned cols)
//{
//	static unsigned need_init = 1;
//	static unsigned *mask;
//	if (need_init) {
//		mask = (unsigned *)calloc(g_p.board_ints * (g_p.board_width-1), sizeof(unsigned));
//		for (int delta = 1; delta < g_p.board_width; delta++) {
//			for (int col = 0; col < delta; col++) {
//				for (int row = 0; row < g_p.board_height; row++) {
//					set_val_for_cell(row, col, mask + (delta-1)*g_p.board_ints, 1);
//				}
//			}
//		}
//	}
//	return mask + (cols-1)*g_p.board_ints;
//}
//
// a mask for the unused bits above the board
//unsigned *mask_rows_top()
//{
//	static unsigned *mask;
//	if (!mask) {
//		mask = (unsigned *)calloc(g_p.board_ints, sizeof(unsigned));
//		if (g_p.board_unused > 0) {
//			mask[g_p.board_ints-1] = - (1 << (32 - g_p.board_unused));
//		}
//	}
//	return mask;
//}
//
// shift the board by a number of columns
//void colshift(unsigned *board, int delta)
//{
//	printf("colshift -- initial board:\n");
//	dump_board(board);
//	
//	if (delta < 0) {
//		delta = -delta;
//
//		printf("shifting the board %d columns to the left\n", delta);
//
//		// shift the low int to the left (lo-bits) first
//		board[0] >>= delta;
//		
//		printf("board after shifting the low int to the left\n");
//		dump_board(board);
//		
//		// get a mask to make the new columns blank
//		unsigned *mask = mask_cols_right(delta);
//
//		if (g_p.board_ints > 1) {
//			// get the bits that will shift out of the hi int
//			unsigned bits = ((1 << delta) - 1) & board[1];
//			// align them to the highest bits
//			bits <<= (32 - delta);
//			// add them to the lo int
//			board[0] |= bits;
//			// now shift the hi int by delta
//			board[1] >>= delta;
//			// mask off the bits in the new cols
//			board[1] &= ~mask[1];
//		}
//		board[0] &= ~mask[0];
//	}else if (delta > 0) {
//	
//		printf("shifting the board %d columns to the right\n", delta);
//
//		// shift the high int to the right (hi-bits) first
//		unsigned *col_mask = mask_cols_left(delta);
//
//		printf("col_mask:\n");
//		dump_board(col_mask);
//
//		unsigned *row_mask = mask_rows_top();
//
//		if (g_p.board_ints > 1) {
//			board[1] <<= delta;
//		
//			printf("board after shifting the hi int to the right\n");
//			dump_board(board);
//		
//			// get the bits that will shift out of the lo int
//			unsigned bits = -(1 << (32-delta)) & board[0];
//			// align them to the lo bits
//			bits >>= 32-delta;
//			// add them to the hi int
//			board[1] |= bits;
//			
//			printf("board after adding the bits that shifted from lo int to hi int\n");
//			dump_board(board);
//			
//			// mask off the rows above the board and cols on the left
//			board[1] &= ((~row_mask[1]) & (~col_mask[1]));
//			
//			printf("board after masking off the hi int\n");
//			dump_board(board);
//		}
//		board[0] <<= delta;
//
//		printf("board after shifting the lo int to the right\n");
//		dump_board(board);
//
//		board[0] &= ~row_mask[0] & ~col_mask[0];
//
//		printf("board after masking off the lo int\n");
//		dump_board(board);
//	}
//}
//
//void rowshift(unsigned *board, int delta)
//{
//	colshift(board, delta * g_p.board_width);
//}
//
//void shift_board(unsigned *board, int colDelta, int rowDelta)
//{
//	colshift(board, colDelta + g_p.board_width * rowDelta);
//}
//
//void shift_state(unsigned *state, int colDelta, int rowDelta)
//{
//	shift_board(state, colDelta, rowDelta);
//	shift_board(state + g_p.board_ints, colDelta, rowDelta);
//}


// copy the starting state to the provided location
void copy_start_state(unsigned *state)
{
	bcopy(g_start_state, state, g_p.state_size * sizeof(unsigned));
}

void switch_sides(unsigned *state)
{
	for (int i = 0; i < g_p.board_size; i++) {
		unsigned temp = X_BOARD(state)[i];
		X_BOARD(state)[i] = O_BOARD(state)[i];
		O_BOARD(state)[i] = temp;
	}
}
void copy_state(unsigned *to, unsigned *from)
{
	for (int i = 0; i < g_p.state_size; i++) {
		to[i] = from[i];
	}
}

unsigned count_pieces(unsigned *board)
{
	unsigned count = 0;
	for (int i = 0; i < g_p.board_size; i++) {
		if (board[i]) ++count;
	}
	return count;
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

//void dump_boards(unsigned *b1, unsigned *b2)
//{
//	unsigned *state = (unsigned *)malloc(2 * g_p.state_size * sizeof(unsigned));
//	bcopy(b1, state, g_p.state_size * sizeof(unsigned));
//	bcopy(b2, state + g_p.state_size, g_p.state_size * sizeof(unsigned));
//	dump_state(state);
//}

void dump_state_ints(unsigned *state)
{
	printf("[STATE]\n");
	for (int i = 0; i < g_p.state_size; i++) {
		printf("%11u", state[i]);
	}
	printf("]\n");
}

void dump_state(unsigned *state, unsigned turn, unsigned nextToPlay)
{
//	printf("dump_state for %u %u %u %u\n", state[3], state[2], state[1], state[0]);
	printf("turn %3d, %s to play:\n", turn, (nextToPlay ? "O" : "X"));
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
//	dump_state_ints(state);
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

	dump_wgts_header("[ WEIGHTS]");
	// get the weight pointer for this agent
	float *pWgts = agCPU->wgts + iag * g_p.num_wgts;
	printf("[    B->H]"); dump_wgts(pWgts);
	for (int i = 0; i < g_p.state_size; i++){
		printf("[IN%03d->H]", i); dump_wgts(pWgts + (1+i) * g_p.num_hidden);
	}
	printf("[    H->O]"); dump_wgts(pWgts + (1+g_p.state_size) * g_p.num_hidden);
	printf("[    B->O], %9.4f\n\n", pWgts[(2+g_p.state_size) * g_p.num_hidden]);

	if (dumpW) {
		dump_wgts_header("[    W    ]");
		// get the W pointer for this agent
		float *pW = agCPU->e + iag * g_p.num_wgts;
		printf("[    B->H]"); dump_wgts(pW);
		for (int i = 0; i < g_p.state_size; i++){
			printf("[IN%03d->H]", i); dump_wgts(pW + (1+i) * g_p.num_hidden);
		}
		printf("[    H->O]"); dump_wgts(pW + (1+g_p.state_size) * g_p.num_hidden);
		printf("[    B->O], %9.4f\n\n", pW[(2+g_p.state_size) * g_p.num_hidden]);
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
	for (int i = 0; i < g_p.state_size; i++){
		printf("[IN%03d->H]", i); dump_wgts(ag->fdata +ag->iWgts + (1+i) * g_p.num_hidden);
	}
	printf("[    H->O]"); dump_wgts(ag->fdata + ag->iWgts + (1+g_p.state_size) * g_p.num_hidden);
	printf("[    B->O], %9.4f\n", ag->fdata[ag->iWgts + (2+g_p.state_size) * g_p.num_hidden]);
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
//void calc_bits_for_board_ints()
//{
//	g_bits_for_board_ints[0] = (g_p.board_size > 32) ? 32 : g_p.board_size;
//	if (g_p.board_ints == 1) {
//		g_bits_for_board_ints[1] = 0;
//	}else {
//		g_bits_for_board_ints[1] = g_p.board_size - 32;
//	}
//}

// calculates agent pointers based on offset from ag->wgts
// agent data is organized as follows
void set_agent_float_pointers(AGENT *ag)
{
	ag->e = ag->wgts + g_p.num_wgts * g_p.num_agents;
	ag->alpha = ag->e + g_p.num_wgts * g_p.num_agents;
	ag->epsilon = ag->alpha + g_p.num_agents;
	ag->lambda = ag->epsilon + g_p.num_agents;
}


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
	g_moves will be of size board_size * 8 * board_size
*/
void build_move_array()
{
	g_moves = (int *)malloc(g_p.board_size * MAX_MOVES * sizeof(int));
	for (int row = 0; row < g_p.board_height; row++) {
		for (int col = 0; col < g_p.board_width; col++) {
			for (int m = 0; m < 8; m++) {
				unsigned iMoves = index4rc(row, col) * MAX_MOVES + m;				
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

AGENT *init_agentsCPU(PARAMS p)
{
	printf("init_agentsCPU...\n");
	// save the parameters and calculate any global constants based on the parameters
	g_p = p;
	build_move_array();
	build_start_state();

	// allocate and initialize the agent data on CPU
	AGENT *ag = (AGENT *)malloc(sizeof(AGENT));

	ag->seeds = (unsigned *)malloc(4 * p.num_agents * sizeof(unsigned));
	for (int i = 0; i < 4*p.num_agents; i++) ag->seeds[i] = rand();
	
	ag->wgts = (float *)malloc(p.num_wgts * p.num_agents * sizeof(float));
	ag->e = (float *)malloc(p.num_wgts * p.num_agents * sizeof(float));
	ag->alpha = (float *)malloc(p.num_agents * sizeof(float));
	ag->epsilon = (float *)malloc(p.num_agents * sizeof(float));
	ag->lambda = (float *)malloc(p.num_agents * sizeof(float));
		
	// initialize values
//	printf("initializing weights for %d values with min of %f and max of %f\n", p.num_wgts * p.num_agents, p.init_wgt_min, p.init_wgt_max);
	for (int i=0; i < p.num_wgts * p.num_agents; i++){
		ag->wgts[i] = rand_wgt2(p.init_wgt_min, p.init_wgt_max);
		ag->e[i] = 0.0f;
	}
	
//	printf("weights and W have been initialized\n");
	
	for (int i = 0; i < p.num_agents; i++) {
		ag->alpha[i] = p.alpha;
		ag->epsilon[i] = p.epsilon;
		ag->lambda[i] = p.lambda;
	}
	
//	printf("alpha, epsilon, and lambda have been initialized\n");
	
	return ag;
}

char *move_string(char *buff, unsigned col, unsigned row)
{
	buff[0] = 'a' + col;
	buff[1] = '1' + row;
	return buff;
}

char *move_stringi(char *buff, unsigned i)
{
	return move_string(buff, i % g_p.board_width, i / g_p.board_width);
}



#pragma mark CPU - run

float random_move(unsigned *state)
{
//	printf("choose_move...\n");
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
				int iTo = g_moves[iFrom * MAX_MOVES + m];
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
	
	unsigned r = move_count * ranf();
	unsigned iRandFrom = possible_moves[r*2];
	unsigned iRandTo = possible_moves[r*2 + 1];
	// do the random move and return the value
	X_BOARD(state)[iRandFrom] = 0;
	X_BOARD(state)[iRandTo] = 1;
	O_BOARD(state)[iRandTo] = 0;
	
//	printf("best move with value %9.4f:\n", bestVal);
//	dump_state(state);
//	printf("\n\n");
	// recalculate to fill in hidden and out for the chosen move
	return 0.0f;
}


/*
	Choose the move for player X from the given state using the nn specified by wgts.
	Return the value of the best state and over-write currState with the best state.
	calculate the values for every possible next state, and put the best one into nextState, returning its value
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
				int iTo = g_moves[iFrom * MAX_MOVES + m];
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
float take_random_action(unsigned *state, unsigned *terminal)
{
	float r = reward(state, terminal);
	if (*terminal) return r;	// given state is terminal, just return the reward
	switch_sides(state);
//	printf("state after switching sides:\n");
//	dump_state(state);
	random_move(state);
//	printf("state after opponent move:\n");
//	dump_state(state);
	switch_sides(state);
//	printf("state after switching sides again:\n");
//	dump_state(state);
	return reward(state, terminal);
}

void set_start_state(unsigned *state, unsigned pieces)
{
//	printf("set_start_state...\n");
	(pieces == 0) ? copy_start_state(state) : random_state(state, pieces);
}

void reset_trace(float *e)
{
	for (int i = 0; i < g_p.num_wgts; i++) {
		e[i] = 0.0f;
	}
}

void update_wgts(float alpha, float delta, float *wgts, float *e)
{
	for (int i = 0; i < g_p.num_wgts; i++) {
		wgts[i] += alpha * delta * e[i];
	}
}

// update eligibility traces using the activation values for hidden and output nodes
void update_trace(unsigned *state, float *wgts, float *e, float *hidden, float *out, float lambda)
{
#ifdef DUMP_MOVES
	printf("update_trace\n");
#endif
	// first decay all existing values
	for (int i = 0; i < g_p.num_wgts; i++) {
		e[i] *= g_p.gamma * lambda;
	}
	
	// next update the weights from hidden layer to output node
	// first the bias
	float g_prime_i = out[0] * (1.0f - out[0]);
//	printf("out[0] is %9.4f and g_prime(out) is %9.4f\n", out[0], g_prime_i);
	unsigned iH2O = (2 * g_p.board_size + 1) * g_p.num_hidden;
	e[iH2O + g_p.num_hidden] += -1.0f * g_prime_i;
	
	// next do all the hidden nodes to output node
	for (int j = 0; j < g_p.num_hidden; j++) {
		e[iH2O + j] += hidden[j] * g_prime_i;
//		printf("hidden node %d, activation is %9.4f, increment to e is %9.4f, new e is %9.4f\n", j, hidden[j], g_prime_i*hidden[j], e[iH2O + j]);
	}
	
	// now update the weights to the hidden nodes
	for (int j = 0; j < g_p.num_hidden; j++) {
		float g_prime_j = hidden[j]*(1.0f - hidden[j]) * wgts[iH2O + j] * g_prime_i;
		// first the bias to the hidden node
		e[j] += -1.0f * g_prime_j;
		
		// then all the input -> hidden values
		for (int k = 0; k < g_p.board_size * 2; k++) {
			if (state[k]) e[(k+1)*g_p.num_hidden + j] += g_prime_j;
		}
	}
}

#define SHOW if(show) 

COMPETE_RESULTS compete(float *ag1_wgts, const char *name1, float *ag2_wgts, const char *name2, unsigned start_pieces, unsigned num_games, unsigned turns_per_game, unsigned show)
{
//	printf("\n=================================================================\n");
//	printf("         %s     vs.     %s\n", name1, name2);
	// play a numbr of games, ag1_wgts vs ag2_wgts
	// set up the starting state
	unsigned *state = (unsigned *)malloc(g_p.state_size * sizeof(unsigned));
	float *hidden = (float *)malloc(g_p.num_hidden * sizeof(float));
	float *out = (float *)malloc(g_p.num_hidden * sizeof(float));
	
	COMPETE_RESULTS cr;
	cr.games = num_games;
	cr.wins = 0;
	cr.losses = 0;
	unsigned turn = 0;
	unsigned game = 0;
	unsigned terminal;
	
	SHOW printf(  "-----------------------------------------------------------------\n");
	SHOW printf("game %d:\n", game);

	set_start_state(state, start_pieces);
	if (ranf() < 0.50f) {
		SHOW printf("New game, O plays first\n");
		SHOW dump_state(state, turn, 1);
		(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal)
					: take_random_action(state, &terminal));
		++turn;
	}else {
		SHOW printf("New game, X plays first\n");
	}
	SHOW dump_state(state, turn, 0);

	
	float V = (ag1_wgts	? choose_move(state, ag1_wgts, hidden, out)
					: random_move(state));
	
	while (game < num_games) {
		SHOW dump_state(state, turn, 1);
		float reward = (ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal)
									: take_random_action(state, &terminal));
		++turn;
		SHOW dump_state(state, turn, 0);
		
		if (terminal){
			if (reward > 0.0f){ ++cr.wins; SHOW printf("*** game won ***\n");}
			if (reward == 0.0f){ ++cr.losses; SHOW printf("*** game lost ***\n");}
		}
		
		if (terminal || (turn == turns_per_game)) {
			SHOW if (!terminal) printf("*** turn limit reached ***\n");
			if (++game < num_games){
				// get ready for next game
				SHOW printf(  "-----------------------------------------------------------------\n");
				SHOW printf("\ngame %d:\n", game);
				turn = 0;
				terminal = 0;
				set_start_state(state, start_pieces);
				if (ranf() < 0.50f) {
					SHOW printf("New game, O plays first\n");
					SHOW dump_state(state, turn, 1);
					(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal)
								: take_random_action(state, &terminal));
					++turn;
					SHOW dump_state(state, turn, 0);
				}else {
					SHOW printf("New game, X plays first\n");
				}
				SHOW dump_state(state, turn, 0);
			}
		}

		float V_prime = (ag1_wgts	? choose_move(state, ag1_wgts, hidden, out)
								: random_move(state));
		V = V_prime;
	}
	printf("[COMPETE],%5d, %8s,%5d, %8s,%5d", num_games, name1, cr.wins, name2, cr.losses);
//	printf("%7d, %7d, %7d\n", wins, losses, wins-losses);
	free(state);
	free(hidden);
	free(out);
	return cr;
}


// run a learning session using agent ag1 against ag2.  ag2 may be NULL which represents a random player
// Start with a random board with start_pieces per side, or the normal starting board if start_pieces is 0
// Top the learning after num_turns turns (for each player)
void auto_learn(AGENT *ag1, float *ag2_wgts, unsigned start_pieces, unsigned num_games, unsigned turns_per_game)
{
	if (!ag1) {
		printf("***ERROR *** random agent can not learn!!!\n");
		exit(-1);
	}
#ifdef DUMP_ALL_AGENT_UPDATES
	dump_agent(ag1, 0, 1);
#endif
	
#ifdef DUMP_MOVES
	printf("auto_learning for %d games, %d turns per game...\n", num_games, turns_per_game);
#endif	
	// set up the starting state
	unsigned *state = (unsigned *)malloc(g_p.state_size * sizeof(unsigned));
	float *hidden = (float *)malloc(g_p.num_hidden * sizeof(float));
	float *out = (float *)malloc(g_p.num_hidden * sizeof(float));
	
	unsigned wins = 0;
	unsigned losses = 0;
	unsigned turn = 0;
	unsigned terminal = 0;
	unsigned game = 0;
	
	set_start_state(state, start_pieces);
	if (ranf() < 0.50f) {
#ifdef DUMP_MOVES
		printf("New game, O to play first...\n");
		dump_state(state, turn, 1);		
#endif
		float r = (ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal) 
							: take_random_action(state, &terminal));
		++turn;
	}else {
#ifdef DUMP_MOVES
		printf("New game, X to play first...\n");
#endif
	}

#ifdef DUMP_MOVES
	dump_state(state, turn, 0);		
#endif

	// choose the action, storing the next state in agCPU->state and returning the value for the next state
	float V = choose_move(state, ag1->wgts, hidden, out);
	
	update_trace(state, ag1->wgts, ag1->e, hidden, out, ag1->lambda[0]);

#ifdef DUMP_ALL_AGENT_UPDATES
	printf("after updating trace...\n");
	dump_agent(ag1, 0, 1);
#endif
	
	// loop over the number of turns
	while (game < num_games) {

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
									: take_random_action(state, &terminal));
		++turn;
#ifdef DUMP_MOVES
		dump_state(state, turn, 0);		
#endif
		
//		printf("after opponent move, turn %d:\n", turn++);
//		dump_state(state);
		
		if (terminal){
			if (reward > 0.0f) ++wins;
			if (reward == 0.0f) ++losses;
#ifdef DUMP_MOVES
			printf("\n\n****** GAME OVER after %d turns with r = %9.4f *******\n", turn, reward);
			printf("record is now %d - %d\n\n\n", wins, losses);
#endif
		}
		
		if (terminal || (turn == turns_per_game)) {
#ifdef DUMP_MOVES
			if (!terminal) printf("****** GAME OVER: reached maximum number of turns per game\n");
#endif
			if (++game < num_games) {
#ifdef DUMP_MOVES
				printf("\n\n--------------- game %d ---------------------\n", game);
#endif
				turn = 0;
				set_start_state(state, start_pieces);
				if (ranf() < 0.50f) {
#ifdef DUMP_MOVES
					printf("New game, O to play first...\n");
					dump_state(state, turn, 1);		
#endif
					(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal) 
								: take_random_action(state, &terminal));
					++turn;
				}else {
#ifdef DUMP_MOVES
					printf("New game, X to play first...\n");
#endif
				}

#ifdef DUMP_MOVES
				dump_state(state, turn, 0);		
#endif
			}
		}
//		printf("choosing next move...\n");
		float V_prime = choose_move(state, ag1->wgts, hidden, out);
		float delta = reward + (terminal ? 0.0f : (g_p.gamma * V_prime)) - V;
//		printf("updating wgts...\n");
		update_wgts(ag1->alpha[0], delta, ag1->wgts, ag1->e);

#ifdef DUMP_ALL_AGENT_UPDATES
		printf("reward = %9.4f, V_prime = %9.4f, V = %9.4f, delta = %9.4f\n", reward, V_prime, V, delta);
		printf("after updating weights:\n");
		dump_agent(ag1, 0, 1);
#endif

		if (terminal) reset_trace(ag1->e);
//		printf("updating trace...\n");
		update_trace(state, ag1->wgts, ag1->e, hidden, out, ag1->lambda[0]);

#ifdef DUMP_ALL_AGENT_UPDATES
		printf("after updating trace:\n");
		dump_agent(ag1, 0, 1);
#endif
		
		V = V_prime;
	}
//	printf("learning session is over...\n");
//	printf("%7d, %7d, %7d\n", wins, losses, wins-losses);
	free(state);
	free(hidden);
	free(out);
}

void copy_agent(AGENT *agCPU, unsigned iFrom, unsigned iTo)
{
	// copy wgts
//	printf("copying agent weights from %d to %d ... ", iFrom, iTo);
	for (int i = 0; i < g_p.num_wgts; i++) {
		agCPU->wgts[i + iTo * g_p.num_wgts] = agCPU->wgts[i + iFrom * g_p.num_wgts];
	}
//	printf("done\n");
}

void progress_indicator(COMPETE_RESULTS after, COMPETE_RESULTS before)
{
//	printf("\nbefore: W%4d  L%4d  Net%4d", before.wins, before.losses, before.wins - before.losses);
//	printf(" after: W%4d  L%4d  Net%4d", after.wins, after.losses, after.wins - after.losses);
	printf("  ,  W:%+4d L:%+4d NET:%+4d", after.wins - before.wins, after.losses - before.losses, (after.wins - after.losses) - (before.wins - before.losses));
}

#define NAME_BUFF_SIZE 16
const char *aname(unsigned n)
{
	static char buff[NAME_BUFF_SIZE];
	snprintf(buff, NAME_BUFF_SIZE, "FT_%d", n);
	return buff;
}

const char *oname(unsigned n)
{
	static char buff[NAME_BUFF_SIZE];
	snprintf(buff, NAME_BUFF_SIZE, "FT_%d", n);
	return buff;
}

RESULTS *runCPU(AGENT *agCPU)
{	
	// test with agent 0
//	dump_agent(agCPU, 0, 0);

	unsigned pieces = 5;
	unsigned max_turns = 20;
	unsigned games_per_rep = 100;
	unsigned reps = 2000;
	if (reps > g_p.num_agents) reps = g_p.num_agents;

	unsigned test_pieces = pieces;
	unsigned test_games = 10;
	unsigned test_max_turns = 20;
	
	COMPETE_RESULTS *results = (COMPETE_RESULTS *)malloc(reps * sizeof(COMPETE_RESULTS));
	
	printf("NODES, BOARD_WIDTH, BOARD_HEIGHT, PIECES, MAX_TURNS, GAMES, REPS\n");
	printf("%d, %d, %d, %d, %d, %d, %d\n", g_p.num_hidden, g_p.board_width, g_p.board_height, pieces, max_turns, games_per_rep, reps);
//	printf("initial agent, before learning:\n");
//	dump_agent(agCPU, 1, 0);
	

	// get a new set of agents and re-run with lambda = 0.0f
	char *base_name = "20TURNS";
	g_p.alpha = 0.20;
	char *mod_name = "10TURNS";
	AGENT *agCPU2 = init_agentsCPU(g_p);


	COMPETE_RESULTS cr;
	COMPETE_RESULTS totcr;

	// first learn against the random agent...
	auto_learn(agCPU, NULL, pieces, games_per_rep, max_turns);
	
	for (int i = 1; i < reps; i++) {
//		totcr.wins = totcr.losses = 0;
		// save the current state of learning agent after i learning sessions into agent slot i [1..reps-1]
		copy_agent(agCPU, 0, i);
//		printf("agent after %d episodes of learning:\n", i);
//		dump_agent(agCPU, i, 0);

		// do a round of competition, first agent vs. RAND
//		printf("[ROUND%3d], ", i);
//		cr = compete(agCPU->wgts, aname(i), NULL, "RAND", test_pieces, test_games, test_max_turns, 0);
//		// print the progress indicator if agent has competed against this agent before
//		if (i > 1){
//			progress_indicator(cr, results[0]);
//			totcr.wins += cr.wins - results[0].wins;
//			totcr.losses += cr.losses - results[0].losses;
//		}
//		printf("\n");
//		
//		// save the results against this agent
//		results[0] = cr;
//
//		// repeat for all the saved agents, not including the one just saved (at i)
//		for (int j = 1; j < i; j++) {
//			printf("[ROUND%3d], ", i);
//			cr = compete(agCPU->wgts, aname(i), agCPU->wgts + j * g_p.num_wgts, oname(j), test_pieces, test_games, test_max_turns, 0);
////			printf("\n            ");
////			cr = compete(agCPU->wgts + j*g_p.num_wgts, oname(j), agCPU->wgts, aname(i), test_pieces, test_games, test_max_turns, 0);
//			if (j < (i-1)){
//				progress_indicator(cr, results[j]);
//				totcr.wins += cr.wins - results[j].wins;
//				totcr.losses += cr.losses - results[j].losses;
//			}
//			printf("\n");
//			results[j] = cr;
//		}
//		printf("                                              total progress is W:%+4d L:%+4d NET:%+4d\n", totcr.wins, totcr.losses, totcr.wins - totcr.losses);
		// do the next round of learning against the last saved agent (at i)
		auto_learn(agCPU, agCPU->wgts + i * g_p.num_wgts, pieces, games_per_rep, max_turns);
	}
	
	// last round of competition
//	printf("[ROUND%3d], ", reps);
//	cr = compete(agCPU->wgts, aname(reps), NULL, "RAND", test_pieces, test_games, test_max_turns, 0);
//	if (reps > 1) progress_indicator(cr, results[0]);
//	printf("\n");
//	results[0] = cr;
	
//	for (int j = 1; j < reps; j++) {
//		printf("[ROUND%3d], ", reps);
//		cr = compete(agCPU->wgts, aname(reps), agCPU->wgts + j * g_p.num_wgts, oname(j), test_pieces, test_games, test_max_turns, 0);
//		printf("\n            ");
//		cr = compete(agCPU->wgts + j*g_p.num_wgts, oname(j), agCPU->wgts, aname(reps), test_pieces, test_games, test_max_turns, 0);
//		if (j < (reps-1)) progress_indicator(cr, results[j]);
//		printf("\n");
//		results[j] = cr;
//	}



	max_turns = 40;



	// first learn against the random agent...
	auto_learn(agCPU2, NULL, pieces, games_per_rep, max_turns);
	
	for (int i = 1; i < reps; i++) {
//		totcr.wins = totcr.losses = 0;
		// save the current state of learning agent after i learning sessions into agent slot i [1..reps-1]
		copy_agent(agCPU2, 0, i);
//		printf("agent after %d episodes of learning:\n", i);
//		dump_agent(agCPU, i, 0);

//		// do a round of competition, first agent vs. RAND
//		printf("[ROUND%3d], ", i);
//		cr = compete(agCPU2->wgts, aname(i), NULL, "RAND", test_pieces, test_games, test_max_turns, 0);
//		// print the progress indicator if agent has competed against this agent before
//		if (i > 1){
//			progress_indicator(cr, results[0]);
//			totcr.wins += cr.wins - results[0].wins;
//			totcr.losses += cr.losses - results[0].losses;
//		}
//		printf("\n");
//		
//		// save the results against this agent
//		results[0] = cr;
//
//		// repeat for all the saved agents, not including the one just saved (at i)
//		for (int j = 1; j < i; j++) {
//			printf("[ROUND%3d], ", i);
//			cr = compete(agCPU2->wgts, aname(i), agCPU2->wgts + j * g_p.num_wgts, oname(j), test_pieces, test_games, test_max_turns, 0);
//			if (j < (i-1)){
//				progress_indicator(cr, results[j]);
//				totcr.wins += cr.wins - results[j].wins;
//				totcr.losses += cr.losses - results[j].losses;
//			}
//			printf("\n");
//			results[j] = cr;
//		}
//		printf("                                              total progress is W:%+4d L:%+4d NET:%+4d\n", totcr.wins, totcr.losses, totcr.wins - totcr.losses);
//		// do the next round of learning against the last saved agent (at i)
		auto_learn(agCPU2, agCPU2->wgts + i * g_p.num_wgts, pieces, games_per_rep, max_turns);
	}
	
	// last round of competition
//	printf("[ROUND%3d], ", reps);
//	cr = compete(agCPU2->wgts, aname(reps), NULL, "RAND", test_pieces, test_games, test_max_turns, 0);
//	if (reps > 1) progress_indicator(cr, results[0]);
//	printf("\n");
//	results[0] = cr;
//	
//	for (int j = 1; j < reps; j++) {
//		printf("[ROUND%3d], ", reps);
//		cr = compete(agCPU2->wgts, aname(reps), agCPU2->wgts + j * g_p.num_wgts, oname(j), test_pieces, test_games, test_max_turns, 0);
//		if (j < (reps-1)) progress_indicator(cr, results[j]);
//		printf("\n");
//		results[j] = cr;
//	}

	test_games = 1000;
	
	// final competition between the two last agents
	compete(agCPU->wgts, base_name, agCPU2->wgts, mod_name, test_pieces, test_games, test_max_turns, 0);
	printf("\n");
	compete(agCPU2->wgts, mod_name, agCPU->wgts, base_name, test_pieces, test_games, test_max_turns, 0);

	
	// show a few games of against the best saved agent
#ifdef SHOW_SAMPLE_GAMES_AFTER_LEARNING
	compete(agCPU->wgts, base_name, agCPU2->wgts, mod_name, pieces, SHOW_SAMPLE_GAMES_AFTER_LEARNING, max_turns, 1);
#endif
	free(results);
	return NULL;
}


#pragma mark -
#pragma mark GPU - Only

AGENT *init_agentsGPU(AGENT *agCPU)
{
	AGENT *agGPU = (AGENT *)malloc(sizeof(AGENT));
	agGPU->seeds = device_copyui(agCPU->seeds, 4 * g_p.num_agents);
	agGPU->wgts = device_copyf(agCPU->wgts, g_p.num_agent_floats * g_p.num_agents);
	set_agent_float_pointers(agGPU);
	
	return agGPU;
}


RESULTS *runGPU(AGENT *agGPU)
{
	return NULL;
}




