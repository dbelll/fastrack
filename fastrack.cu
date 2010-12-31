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

__constant__ unsigned dc_reps_for_wgts;	// number of times must loop to have threads cover all the weights
										// = 1 + (num_wgts-1)/board_size
										
__constant__ int *dc_moves;			// pointer to d_g_moves array on device

__constant__ unsigned dc_max_turns;
__constant__ unsigned dc_episode_length;
__constant__ unsigned dc_benchmark_games;

// prototypes
void freeAgentCPU(AGENT *ag);
void set_start_state(unsigned *state, unsigned pieces, unsigned *seeds, unsigned stride);
void dump_all_wgts(float *wgts, unsigned num_hidden);

#pragma mark -
#pragma mark Helpers

// return a random number from 0 to nn-1
__device__ __host__ unsigned rand_num(unsigned nn, unsigned *seeds, unsigned stride)
{
	unsigned r = RandUniformui(seeds, stride) % nn;
	return r;
}

/*
	Accessors for the wgts (and e's) 
	There are separate accessors for global memory and shared memory.  In global memory,
	the weights are aligned to 128 byte boundaries for faster access.  In shared memory, the
	weights are stored more compactly.
	Weight names:
		BH	bias -> hidden
		IH	input -> hidden
		HO	hidden -> output
		BO	bias -> output
*/
//float *pBH(float *w, unsigned iH){ return w + iH; }			// bias to hidden
//float *pIH(float *w, unsigned iI, unsigned iH){ return w + iH + g_p.num_hidden * (iI + 1);}	// in -> hidden
//float *pHO(float *w, unsigned iH){ return w + g_p.num_hidden * (1 + g_p.state_size) + iH; } // hidden -> out
//float *pBO(float *w){ return w + g_p.num_wgts - g_p.num_hidden; }	// bias to output

#if GLOBAL_WGTS_FORMAT == 1
/*
	version 1 - compact storage for global and shared memory
		B->H	(num_hidden)
		I->H	(state_size * num_hidden)
		H->O	(num_hidden)
		B->O	1
	Weights padded out to be (state_size + 3) * num_hidden values for both shared and global.
*/
float *pgBH(float *w, unsigned iH){ return w + iH; }
float *pgIH(float *w, unsigned iI, unsigned iH){ return w + iH + g_p.num_hidden * (iI + 1);}
float *pgHO(float *w, unsigned iH){ return w + g_p.num_hidden * (1 + g_p.state_size) + iH; }
float *pgBO(float *w){ return w + g_p.wgts_stride - g_p.num_hidden; }

float *psBH(float *w, unsigned iH){ return w + iH; }
float *psIH(float *w, unsigned iI, unsigned iH){ return w + iH + g_p.num_hidden * (iI + 1);}
float *psHO(float *w, unsigned iH){ return w + g_p.num_hidden * (1 + g_p.state_size) + iH; }
float *psBO(float *w){ return w + g_p.wgts_stride - g_p.num_hidden; }

#elif GLOBAL_WGTS_FORMAT == 2
/*
	version 2 - global has I->H first with rows corresponding to each hidden node and colums
				padded out to 2 * MAX_STATE_SIZE.  The two boards start at offset 0 and MAX_BOARD_SIZE.
				The B->H and H->O values are next and each is padded out to 2 * MAX_STATE_SIZE columns
				The single value B->O is in the last value of the H->O area.
		I->H	(num_hidden * MAX_STATE_SIZE)
		B->H	(MAX_STATE_SIZE)
		H->O	(MAX_STATE_SIZE-1)
		B->O	1
	Global weights take up (MAX_STATE_SIZE + 2) * num_hidden + 1 values.
	
	Compact format removes all padding:
		I->H	(num_hidden * state_size)
		B->H	num_hidden
		H->O	num_hidden
		B->O	1
	Total size for compact format is (state_size + 2)*num_hidden + 1	
*/

float *pgIXH(float *w, unsigned iX, unsigned iH){ return w + iH * MAX_STATE_SIZE + iX; }
float *pgIOH(float *w, unsigned iO, unsigned iH){ return w + iH * MAX_STATE_SIZE + MAX_BOARD_SIZE + iO; }
float *pgIH(float *w, unsigned iI, unsigned iH){ 
	return (iI >= g_p.board_size) ? pgIOH(w, iI-g_p.board_size, iH) : pgIXH(w, iI, iH); 
}
float *pgBH(float *w, unsigned iH){ return w + g_p.num_hidden * MAX_STATE_SIZE + iH; }
float *pgHO(float *w, unsigned iH){ return w + (1 + g_p.num_hidden) * MAX_STATE_SIZE + iH; }
float *pgBO(float *w){ return w + g_p.wgts_stride - 1; }

float *psIH(float *w, unsigned iI, unsigned iH){ return w + iH * g_p.state_size + iI; }
float *psBH(float *w, unsigned iH){ return w + g_p.state_size * g_p.num_hidden + iH; }
float *psHO(float *w, unsigned iH){ return w + g_p.state_size * (g_p.num_hidden + 1) + iH; }
float *psBO(float *w){ return w + g_p.num_wgts - 1; }

// macros for use on device
// They evaluate to a pointer to the first value for the specified section of weights
#define G_IXH(w, iH) (w + iH * MAX_STATE_SIZE)
#define G_IOH(w, iH) (G_IXH(w, iH) + MAX_BOARD_SIZE)
#define G_BH(w) (w + dc_num_hidden * MAX_STATE_SIZE)
#define G_HO(w) (G_BH(w) + MAX_STATE_SIZE)
#define G_BO(w) (G_HO(w) + MAX_STATE_SIZE - 1)

#define S_IXH(w, iH) (w + iH * dc_state_size)
#define S_IOH(w, iH) (S_IXH(w, iH) + dc_board_size)
#define S_BH(w) (w + dc_num_hidden * dc_state_size)
#define S_HO(w) (S_BH(w) + dc_num_hidden)
#define S_BO(w) (S_HO(w) + dc_num_hidden)


#endif

float gBH(float *w, unsigned iH){ return *pgBH(w, iH);}
float gIH(float *w, unsigned iI, unsigned iH){ return *pgIH(w, iI, iH); }
float gHO(float *w, unsigned iH){ return *pgHO(w, iH); }
float gBO(float *w){ return *pgBO(w); }
float sBH(float *w, unsigned iH){ return *psBH(w, iH);}
float sIH(float *w, unsigned iI, unsigned iH){ return *psIH(w, iI, iH); }
float sHO(float *w, unsigned iH){ return *psHO(w, iH); }
float sBO(float *w){ return *psBO(w); }

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

// calculate index number for given row and column
unsigned index4rc(unsigned row, unsigned col)
{
	return row * g_p.board_width + col;
}

// generate string for the cell at row and column
char *move_string(char *buff, unsigned col, unsigned row)
{
	buff[0] = 'a' + col;
	buff[1] = '1' + row;
	return buff;
}

// generate string for the cell at given index
char *move_stringi(char *buff, unsigned i)
{
	return move_string(buff, i % g_p.board_width, i / g_p.board_width);
}



#pragma mark -
#pragma mark allocating and freeing

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

unsigned game_won(float reward){
	return reward > 0.5f;
}

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

char char_for_board(unsigned row, unsigned col, unsigned *board)
{
	unsigned index = index4rc(row, col);
//	printf("index for row %d, col%d is %d\n", row, col, index);
//	printf("value of board there is %d\n", board[index]);
	char ch = board[index] ? 'X' : '.';
//	printf("char_for_board row=%d, col=%d is %c\n", row, col, ch);
	return ch;
}

char char_for_cell(unsigned row, unsigned col, unsigned *state)
{
	return char_for_index(index4rc(row, col), state);
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

// duplicate the GPU method of getting a random board
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
#pragma mark Read/write stuff

int wl_compare(const void *p1, const void *p2)
{
	const WON_LOSS *wl1 = (const WON_LOSS *)p1;
	const WON_LOSS *wl2 = (const WON_LOSS *)p2;
	float score1 = (wl1->wins - wl1->losses) / (float)wl1->games;
	float score2 = (wl2->wins - wl2->losses) / (float)wl2->games;
	int result = 0;
	if (score1 > score2) result = -1;
	if (score1 < score2) result = 1;
//	printf("(%d-%d-%d  %5.3f) vs (%d-%d-%d  %5.3f) comparison is %d\n", wl1->wins, wl1->losses, wl1->games - wl1->wins - wl1->losses, score1, wl1->wins, wl2->losses, wl2->games - wl2->wins - wl2->losses, score2, result);
	return result; 
}

int wl_byagent(const void *p1, const void *p2)
{
	const WON_LOSS *wl1 = (const WON_LOSS *)p1;
	const WON_LOSS *wl2 = (const WON_LOSS *)p2;
	int result = 0;
	if (wl1->agent < wl2->agent) result = -1;
	if (wl1->agent > wl2->agent) result = 1;
	return result; 
}

// calculate the winning percentage based on games, wins and losses in WON_LOSS structure
float winpct(WON_LOSS wl)
{
	return 0.5f * (1.0f + (float)(wl.wins - wl.losses)/(float)wl.games);
}

// Print sorted standings using the WON_LOSS information in standings (from learning vs. peers),
// and vsChamp (from competing against benchmark agent).
void print_standings(WON_LOSS *standings, WON_LOSS *vsChamp)
{
		qsort(standings, g_p.num_agents, sizeof(WON_LOSS), wl_compare);
		printf(    "             G    W    L    PCT");
		printf("   %4d games vs Champ\n", g_p.benchmark_games);

		WON_LOSS totChamp = {0, 0, 0, 0};
		WON_LOSS totStand = {0, 0, 0, 0};
		
		for (int i = 0; i < g_p.num_agents; i++) {
//			printf("agent%4d  %4d %4d %4d  %5.3f", standings[i].agent, standings[i].games, standings[i].wins, standings[i].losses, 0.5f * (1.0f + (float)(standings[i].wins - standings[i].losses) / (float)standings[i].games));
			printf("agent%4d  %4d %4d %4d  %5.3f", standings[i].agent, standings[i].games, standings[i].wins, standings[i].losses, winpct(standings[i]));
			
			totStand.games += standings[i].games;
			totStand.wins += standings[i].wins;
			totStand.losses += standings[i].losses;
			
			printf("  (%4d-%4d)    %+5d\n", vsChamp[standings[i].agent].wins,vsChamp[standings[i].agent].losses, vsChamp[standings[i].agent].wins - vsChamp[standings[i].agent].losses);
			totChamp.games += vsChamp[standings[i].agent].games;
			totChamp.wins += vsChamp[standings[i].agent].wins;
			totChamp.losses += vsChamp[standings[i].agent].losses;
		}
		printf(" avg      %5d%5d%5d  %5.3f (%5.1f-%5.1f)   %+5.1f\n", totStand.games, totStand.wins, totStand.losses, winpct(totStand), (float)totChamp.wins / (float)g_p.num_agents, (float)totChamp.losses / (float)g_p.num_agents, (float)(totChamp.wins-totChamp.losses) / (float)g_p.num_agents);
}

// write the global parameters to a .CSV file
void save_parameters(FILE *f)
{
	fprintf(f, "SEED, %u, CHAMP, %s\n", g_p.seed, AGENT_FILE_CHAMP);
	fprintf(f, "board size, %d, %d\n", g_p.board_width, g_p.board_height);
	fprintf(f, "NUM_PIECES, %d\n", g_p.num_pieces);
	fprintf(f, "MAX_TURNS, %d\n", g_p.max_turns);
	fprintf(f, "G_ALLOWABLE_MOVES:\n");
	for (int m = 0; m < MAX_MOVES; m++) {
		fprintf(f, "%d, %d\n", g_allowable_moves[m][0], g_allowable_moves[m][1]);
	}
	fprintf(f, "NUM_HIDDEN, %d\n", g_p.num_hidden);
	fprintf(f, "INIT_WGT_MIN and MAX, %9.6f, %9.6f\n", g_p.init_wgt_min, g_p.init_wgt_max);
	fprintf(f, "NUM_AGENTS, %d\n", g_p.num_agents);
	fprintf(f, "NUM_SESSIONS, %d\n", g_p.num_sessions);
	fprintf(f, "EPISODE_LENGTH, %d\n", g_p.episode_length);
	fprintf(f, "WARMUP_LENGTH, %d\n", g_p.warmup_length);
	fprintf(f, "alpha, %9.6f\n", g_p.alpha);
	fprintf(f, "epsilon, %9.6f\n", g_p.epsilon);
	fprintf(f, "gamma, %9.6f\n", g_p.gamma);
	fprintf(f, "lambda, %9.6f\n", g_p.lambda);
}

// write agent weights to a file, including some game parameters
void save_agent(const char *file, AGENT *ag, unsigned iAg)
{
	FILE *f = fopen(file, "w");
	if (!f) {
		printf("Can not open file for saving agent %s\n", file);
		return;
	}
	// current file version number
	static unsigned version = FILE_FORMAT;
	
	fprintf(f, "%d\n", version);
	fprintf(f, "%d\n", g_p.board_width);
	fprintf(f, "%d\n", g_p.board_height);
	fprintf(f, "%d\n", g_p.num_pieces);
	fprintf(f, "%d\n", g_p.max_turns);
	fprintf(f, "%d\n", g_p.num_hidden);
	fprintf(f, "%d\n", g_p.num_wgts);
//	for (int i = 0; i < g_p.wgts_stride; i++) {
//		fprintf(f, "%f\n", ag->wgts[iAg * g_p.wgts_stride + i]);
//	}
	
	// B->H
	for (int iH = 0; iH < g_p.num_hidden; iH++) fprintf(f, "%f\n", gBH(ag->wgts, iH));
	// I->H
	for (int iI = 0; iI < g_p.state_size; iI++)
		for (int iH = 0; iH < g_p.num_hidden; iH++) 
			fprintf(f, "%f\n", gIH(ag->wgts, iI, iH));
	// H->O
	for (int iH = 0; iH < g_p.num_hidden; iH++) fprintf(f, "%f\n", gHO(ag->wgts, iH));
	// B->O
	fprintf(f, "%f\n", gBO(ag->wgts));
	
	fclose(f);
}

// Read weights from the current file position and store in global format at wgts
// wgts must have been allocated to hold g_p.wgts_stride values
// Board width, height, and number of hidden nodes must match global parameters
// File format for weights is:
//		B->H		(num_hidden)
//		I->H		(state_size * num_hidden)
//		H->O		(num_hidden)
//		B->O		1
void read_wgts_aux(FILE *f, float *wgts, unsigned version, unsigned w, unsigned h, unsigned num_hidden)
{
	if (w != g_p.board_width || h != g_p.board_height || num_hidden != g_p.num_hidden) {
		printf("*** ERROR *** agent file does not match current parameters:\n");
		printf("            parameters  agent file\n");
		printf("board size   (%3dx%3d)   (%3dx%3d)\n", g_p.board_width, g_p.board_height, w, h);
		printf("hidden nodes    %3d         %3d\n", g_p.num_hidden, num_hidden);
		return;
	}
	
	switch (version) {
		case 1:
			for (int iH = 0; iH < g_p.num_hidden; iH++) fscanf(f, "%f", pgBH(wgts, iH));
			for (int iI = 0; iI < g_p.state_size; iI++)
				for (int iH = 0; iH < g_p.num_hidden; iH++)
					fscanf(f, "%f", pgIH(wgts, iI, iH));
			for (int iH = 0; iH < g_p.num_hidden; iH++) fscanf(f, "%f", pgHO(wgts, iH));
			fscanf(f, "%f", pgBO(wgts));
			break;
		default:
			printf("*** ERROR *** unknown file format %d\n", version);
			break;
	}
}

// Read an agent's values from a file.  AGENT structure and all members are pre-allocated.
//void read_agent(FILE *f, AGENT *ag)
//{
//	unsigned version;
//	unsigned board_width, board_height;
//	unsigned num_pieces;
//	unsigned max_turns;
//	unsigned num_hidden;
//	unsigned num_wgts;
//	
//	fscanf(f, "%d", &version);
//
//	switch (version) {
//		case 1:
//			fscanf(f, "%u", &board_width);
//			fscanf(f, "%u", &board_height);
//			fscanf(f, "%u", &num_pieces);
//			fscanf(f, "%u", &max_turns);
//			fscanf(f, "%u", &num_hidden);
//			fscanf(f, "%u", &num_wgts);
//			read_wgts_aux(f, ag->wgts, version, board_width, board_height, num_hidden);		
//			break;
//		default:
//			printf("*** ERROR *** unknown file format %d\n", version);
//			break;
//	}
//}
//
//unsigned read_num_wgts(FILE *f)
//{
//	unsigned version;
//	unsigned board_width, board_height;
//	unsigned num_pieces;
//	unsigned max_turns;
//	unsigned num_hidden;
//	unsigned wgts_stride;
//	fseek(f, 0L, SEEK_SET);
//	fscanf(f, "%d", &version);
//
//	switch (version) {
//		case 1:
//			fscanf(f, "%u", &board_width);
//			fscanf(f, "%u", &board_height);
//			fscanf(f, "%u", &num_pieces);
//			fscanf(f, "%u", &max_turns);
//			fscanf(f, "%u", &num_hidden);
//			fscanf(f, "%u", &wgts_stride);
//			break;
//		default:
//			break;
//	}
//	return wgts_stride;
//}
//
//unsigned read_num_hidden(FILE *f)
//{
//	unsigned version;
//	unsigned board_width, board_height;
//	unsigned num_pieces;
//	unsigned max_turns;
//	unsigned num_hidden;
//	unsigned wgts_stride;
//	fseek(f, 0L, SEEK_SET);
//	fscanf(f, "%d", &version);
//
//	switch (version) {
//		case 1:
//			fscanf(f, "%u", &board_width);
//			fscanf(f, "%u", &board_height);
//			fscanf(f, "%u", &num_pieces);
//			fscanf(f, "%u", &max_turns);
//			fscanf(f, "%u", &num_hidden);
//			fscanf(f, "%u", &wgts_stride);
//			break;
//		default:
//			break;
//	}
//	printf("read_num_hidden: (%dx%d) board, %d pieces, %d turns, %d num_hidden, %d wgts_stride\n", board_width, board_height, num_pieces, max_turns, num_hidden, wgts_stride);
//	return num_hidden;
//}

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
#endif

// Read in just the weights from an agent file
// Print a warning if the file does not match global parameters
// wgts must be pre-allocated to hold wgts_stride values
void read_wgts(FILE *f, float *wgts)
{
	unsigned version;
	unsigned board_width, board_height;
	unsigned num_pieces;
	unsigned max_turns;
	unsigned num_hidden;
	unsigned num_wgts;
	
	fseek(f, 0L, SEEK_SET);
	fscanf(f, "%d", &version);

	switch (version) {
		case 1:
			fscanf(f, "%u", &board_width);
			fscanf(f, "%u", &board_height);
			fscanf(f, "%u", &num_pieces);
			fscanf(f, "%u", &max_turns);
			fscanf(f, "%u", &num_hidden);
			fscanf(f, "%u", &num_wgts);

			printf("values from agent file:\n");
			printf("board size: (%3dx%3d)\n", board_width, board_height);
			printf("num_pieces = %d, max_turns = %d\n", num_pieces, max_turns);
			printf("num_hidden = %d, num_wgts = %d\n", num_hidden, num_wgts);

			read_wgts_aux(f, wgts, version, board_width, board_height, num_hidden);
			break;
			
		default:
			break;
	}
	printf("read_wgts: (%dx%d) board, %d pieces, %d turns, %d num_hidden, %d wgts_stride\n", board_width, board_height, num_pieces, max_turns, num_hidden, num_wgts);
}

// load weights for champ from a file
float *load_champ(const char *file)
{
	FILE *f = fopen(file, "r");
	if (!f) {
		printf("could not open champ file %s\n", file);
		exit(-1);
	}
	
	// allocate an array and read in the weights
	float *wgts = (float *)malloc(g_p.wgts_stride * sizeof(float));
	read_wgts(f, wgts);
	
#ifdef DUMP_CHAMP_WGTS
	printf("\n----------------\nchamp weights:\n-----------------\n");
	dump_all_wgts(wgts, g_p.num_hidden);
#endif

	fclose(f);
	return wgts;
}

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
//	printf("state located at %p\n", state);
	printf("\nturn %3d, %s to play:\n", turn, (nextToPlay ? "O" : "X"));
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
}

// dump the board, using X for pieces
void dump_board(unsigned *board)
{
	dump_col_header(3, g_p.board_width);
	for (int row = g_p.board_height - 1; row >= 0; row--) {
		printf("%2u ", row+1);
		for (int col = 0; col < g_p.board_width; col++) {
			printf(" %c", char_for_board(row, col, board));
		}
		printf("%3u", row+1);
		printf("\n");
	}
	dump_col_header(3, g_p.board_width);
}

void dump_boards(unsigned *board, unsigned n)
{
	for (int i = 0; i < n; i++) {
		printf("board %d:\n", i);
		dump_board(board + i * g_p.board_size);
	}
}

void dump_states(unsigned *state, unsigned n)
{
	for (int i = 0; i < n; i++) {
		printf("state %d:\n", i);
		dump_state(state + i * g_p.state_size, 0, 0);
	}
}

void dump_boardsGPU(unsigned *board, unsigned n)
{
	printf("dumping %d boards at %p\n", n, board);
	
	unsigned *boardsCPU = host_copyui(board, n * g_p.board_size);
	host_dumpui("boards copied to host", boardsCPU, n * g_p.board_height, g_p.board_width);
	dump_boards(boardsCPU, n);
	free(boardsCPU);
}

void dump_agentsGPU(const char *str, AGENT *agGPU, unsigned dumpW)
{
	printf("dump_agentsGPU\n");
	
	// create a CPU copy of the GPU agent data
	AGENT *agCPU = (AGENT *)malloc(sizeof(AGENT));
	
	agCPU->seeds = host_copyui(agGPU->seeds, 4 * g_p.num_agents * g_p.board_size);
	agCPU->states = host_copyui(agGPU->states, g_p.num_agents * g_p.state_size);
	agCPU->next_to_play = host_copyui(agGPU->next_to_play, g_p.num_agents);
	agCPU->wgts = host_copyf(agGPU->wgts, g_p.num_agent_floats * g_p.num_agents);
	set_agent_float_pointers(agCPU);
	
	dump_agentsCPU(str, agCPU, dumpW);
	
	freeAgentCPU(agCPU);
}

void dump_statesGPU(unsigned *state, unsigned n)
{
	printf("dumping %d states at %p\n", n, state);
	
	unsigned *statesCPU = host_copyui(state, n * g_p.state_size);
	host_dumpui("states copied to host", statesCPU, 2 * n * g_p.board_height, g_p.board_width);
	dump_states(statesCPU, n);
	free(statesCPU);
}

void dump_wgts_header(const char *str)
{
	printf("%s", str);
	for (int i = 0; i < g_p.num_hidden; i++) {
		printf(",  %6d  ", i);
	}
	printf("\n");
}

// dump of weight values from input to hidden
void dump_wgts_IH(float *wgts, unsigned iI)
{
	printf("[IN%03d->H]", iI); 
	for (int iH = 0; iH < g_p.num_hidden; iH++) {
		printf(", %9.4f", gIH(wgts, iI, iH));
	}
	printf("\n");
}

void dump_wgts_BH(float *wgts)
{
	printf("[    B->H]"); 
	for (int i = 0; i < g_p.num_hidden; i++) {
		printf(", %9.4f", gBH(wgts, i));
	}
	printf("\n");
}

void dump_wgts_HO(float *wgts)
{
	printf("[    H->O]"); 
	for (int i = 0; i < g_p.num_hidden; i++) {
		printf(", %9.4f", gHO(wgts, i));
	}
	printf("\n");
}

void dump_wgts_BO(float *wgts)
{
	printf("[    B->O], %9.4f\n\n", gBO(wgts));
}

// print out all weights in a formatted output
void dump_all_wgts(float *wgts, unsigned num_hidden)
{
	dump_wgts_header("[ WEIGHTS]");
	dump_wgts_BH(wgts);
	for (int i = 0; i < g_p.state_size; i++) dump_wgts_IH(wgts, i);
	dump_wgts_HO(wgts);
	dump_wgts_BO(wgts);
}

void dump_agent(AGENT *agCPU, unsigned iag, unsigned dumpW)
{
#ifndef AGENT_DUMP_BOARD_ONLY
	printf("[SEEDS], %10u, %10u %10u %10u\n", agCPU->seeds[iag], agCPU->seeds[iag + g_p.num_agents], agCPU->seeds[iag + 2 * g_p.num_agents], agCPU->seeds[iag + 3 * g_p.num_agents]);

	dump_wgts_header("[ WEIGHTS]");
	// get the weight pointer for this agent
	float *pWgts = agCPU->wgts + iag * g_p.wgts_stride;

	dump_wgts_BH(pWgts);
	for (int i = 0; i < g_p.state_size; i++) dump_wgts_IH(pWgts, i);
	dump_wgts_HO(pWgts);
	dump_wgts_BO(pWgts);

	if (dumpW) {
		dump_wgts_header("[    W    ]");
		// get the W pointer for this agent
		float *pW = agCPU->e + iag * g_p.wgts_stride;
		dump_wgts_BH(pW);
		for (int i = 0; i < g_p.state_size; i++) dump_wgts_IH(pW, i);
		dump_wgts_HO(pW);
		dump_wgts_BO(pW);
	}

	printf("[   alpha], %9.4f\n", agCPU->alpha[iag]);
	printf("[ epsilon], %9.4f\n", agCPU->epsilon[iag]);
	printf("[  lambda], %9.4f\n", agCPU->lambda[iag]);
#endif	
	dump_state(agCPU->states + iag*g_p.state_size, 0, agCPU->next_to_play[iag]);
}

// dump all agents, flag controls if the eligibility trace values are also dumped
void dump_agentsCPU(const char *str, AGENT *agCPU, unsigned dumpW)
{
	printf("======================================================================\n");
	printf("%s\n", str);
	printf("----------------------------------------------------------------------\n");
	for (int i = 0; i < g_p.num_agents; i++) {
		printf("\n[AGENT%5d] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n", i);
		dump_agent(agCPU, i, dumpW);
	}
	printf("======================================================================\n");
	
}


#pragma mark -
#pragma mark RESULTS functions
void dumpResults(RESULTS *r)
{
	FILE *f = fopen(LEARNING_LOG_FILE, "w");
	if (!f) {
		printf("could not open the LEARNING_LOG_FILE %s\n", LEARNING_LOG_FILE);
		return;
	}

	save_parameters(f);
	
	// loop through each session and save the won-loss data to the LEARNING_LOG_FILE
	for (int iSession = 0; iSession < g_p.num_sessions; iSession++) {
		// sort the standings by agent number (vsChamp is already by agent number)
		qsort(r->standings + iSession * g_p.num_agents, g_p.num_agents, sizeof(WON_LOSS), wl_byagent);
		for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
			unsigned iStand = iSession * g_p.num_agents + iAg;
			fprintf(f, "%d, %d, %d, %d, %d, %d, %d, %d\n", iSession, iAg, r->standings[iStand].games, r->standings[iStand].wins, r->standings[iStand].losses, r->vsChamp[iStand].games, r->vsChamp[iStand].wins, r->vsChamp[iStand].losses);
		}
	}
	fclose(f);
}

// Allocate a RESULTS struture to hold results for p.num_sessions
RESULTS *newResults(PARAMS p)
{
	RESULTS *results = (RESULTS *)malloc(sizeof(RESULTS));
	results->p = p;
	results->standings = (WON_LOSS *)malloc(p.num_sessions * p.num_agents * sizeof(WON_LOSS));
	results->vsChamp = (WON_LOSS *)malloc(p.num_sessions * p.num_agents * sizeof(WON_LOSS));
	return results;
}


void freeResults(RESULTS *r)
{
	if (r) {
		if(r->standings) free(r->standings);
		if(r->vsChamp) free(r->vsChamp);
		free(r);
	}
}


#pragma mark Initialization
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
	
	printf("agGPU->seeds %p\n", agGPU->seeds);
	printf("agGPU->states %p\n", agGPU->states);
	printf("agGPU->wgts %p\n", agGPU->wgts);
	
	int *d_g_moves = device_copyi(g_moves, g_p.board_size * MAX_MOVES);
	
//	host_dumpi("g_moves", g_moves, MAX_MOVES, g_p.board_size);
//	device_dumpi("d_moves", d_g_moves, MAX_MOVES, g_p.board_size);
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_moves", &d_g_moves, sizeof(int *)));
	
	// copy parameter values to constant memory
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_ag", agGPU, sizeof(AGENT)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_gamma", &g_p.gamma, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_hidden", &g_p.num_hidden, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_pieces", &g_p.num_pieces, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_state_size", &g_p.state_size, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_board_size", &g_p.board_size, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_num_wgts", &g_p.num_wgts, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_wgts_stride", &g_p.wgts_stride, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_half_board_size", &g_p.half_board_size, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_half_hidden", &g_p.half_hidden, sizeof(unsigned)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_max_turns", &g_p.max_turns, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_episode_length", &g_p.episode_length, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("dc_benchmark_games", &g_p.benchmark_games, sizeof(unsigned)));
	
//	unsigned half_board_size = 1;
//	while (half_board_size < g_p.board_size) {
//		half_board_size <<= 1;
//	}
//	half_board_size >>=1;	// now half_board_size is the largest power of two less board_size
	
	unsigned reps_for_wgts = 1 + (g_p.num_wgts - 1) / g_p.board_size;
	printf("reps_for_wgts on GPU is %d\n", reps_for_wgts);
	cudaMemcpyToSymbol("dc_reps_for_wgts", &reps_for_wgts, sizeof(unsigned));
	
	float piece_ratioX = (float)g_p.num_pieces / (float)g_p.board_size;
	float piece_ratioO = (float)g_p.num_pieces / (float)(g_p.board_size - g_p.num_pieces);
	cudaMemcpyToSymbol("dc_piece_ratioX", &piece_ratioX, sizeof(float));
	cudaMemcpyToSymbol("dc_piece_ratioO", &piece_ratioO, sizeof(float));
	printf("device constants:\n");
	printf("   dc_num_hidden: %4d\n", g_p.num_hidden);
	printf("   dc_num_pieces: %4d\n", g_p.num_pieces);
	printf("   dc_state_size: %4d\n", g_p.state_size);
	printf("   dc_board_size: %4d\n", g_p.board_size);
	printf("   dc_num_wgts:   %4d\n", g_p.num_wgts);
	printf("   dc_piece_ratioX: %9.6f\n", piece_ratioX);
	printf("   dc_piece_ratioO: %9.6f\n", piece_ratioO);
	
	return agGPU;
}


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
	
	unsigned r = move_count * RandUniform(seeds, stride);
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
	unsigned iH2O = (2 * g_p.board_size + 1) * g_p.num_hidden;
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
//	set_start_state(state, start_pieces, seeds, g_p.board_size);

//	return wl;
	
#ifdef DUMP_MOVES
	printf("--------------- game %d ---------------------\n", wl.games);
#endif

//	if (RandUniform(seeds, g_p.board_size) < 0.50f) {
	if (ag1->next_to_play[iAg]) {

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
	printf("running on CPU...\n");

#ifdef DUMP_INITIAL_AGENTS
	dump_agentsCPU("initial agents on CPU", agCPU, 1);
#endif

//	for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
//		auto_learn(agCPU, iAg, NULL, g_p.num_pieces, g_p.warmup_length, g_p.max_turns, 0);
//	}
	
//	return NULL;
	
	// allocate the structure to hold the results
	RESULTS *r = newResults(g_p);
	
	// lastWinner is the agent in first place after each learning session
	unsigned lastWinner = 0;

	if (g_p.warmup_length > 0) {
		printf("warm-up versus RAND\n");
//		dump_agentsCPU("prior to warmup", agCPU, 0);
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

	for (int iSession = 0; iSession < g_p.num_sessions; iSession++) {
		// copy the current weights to the saved_wgts area
		memcpy(agCPU->saved_wgts, agCPU->wgts, g_p.num_agents * g_p.wgts_stride * sizeof(float));
		
		printf("\n********** Session %d **********\n", iSession);
//		printf("g_p.num_hidden is %d\n", g_p.num_hidden);
		// run a round-robin learning session
		for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
			unsigned iStand = iSession * g_p.num_agents + iAg;
			r->standings[iStand].agent= iAg;
			r->standings[iStand].games = 0;
			r->standings[iStand].wins = 0;
			r->standings[iStand].losses = 0;

//			for (int iOp = 0; iOp < g_p.num_agents; iOp++) {
//				unsigned xOp = (iAg + iOp) % g_p.num_agents;

			// compete against the top half of the agents from previous standings
//			for (int iOp = 0; iOp < g_p.num_agents/((iSession > 0) ? 2 : 1); iOp++) {
			for (int iOp = 0; iOp < 1; iOp++) {		// for debugging, just run vs. one opponent
				unsigned xOp = (iAg + iOp) % g_p.num_agents;
				if (iSession > 0) xOp = r->standings[(iSession-1) * g_p.num_agents + iOp].agent;

//				printf("\n\n>>>>> new matchup >>>>> (%d vs %d)\n", iAg, xOp);

				WON_LOSS wl = auto_learn(agCPU, iAg, agCPU->saved_wgts + xOp * g_p.wgts_stride, g_p.num_pieces, g_p.episode_length, g_p.max_turns);
//				printf("g_p.num_hidden is %d\n", g_p.num_hidden);
				r->standings[iStand].games += wl.games;
				r->standings[iStand].wins += wl.wins;
				r->standings[iStand].losses += wl.losses;
			}

			// just compete against the champ
//			WON_LOSS wl = auto_learn(agCPU, iAg, champ_wgts, g_p.num_pieces, g_p.episode_length, g_p.max_turns, g_p.num_hidden);
//			r->standings[iStand].games += wl.games;
//			r->standings[iStand].wins += wl.wins;
//			r->standings[iStand].losses += wl.losses;
			
			// compete against the champ as a benchmark
			if (g_p.benchmark_games > 0) {
				r->vsChamp[iStand] = compete(agCPU->wgts + iAg * g_p.wgts_stride, NULL, champ_wgts, NULL, g_p.num_pieces, g_p.benchmark_games, g_p.max_turns, 0, g_p.num_hidden, agCPU->seeds + iAg, g_p.num_agents);
			}
		}
		
		// sort and print the standings
		print_standings(r->standings + iSession * g_p.num_agents, r->vsChamp + iSession * g_p.num_agents);
		
		// remember the winner after this session
		lastWinner = r->standings[iSession * g_p.num_agents + 0].agent;
	}
	
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
	dump_agentsCPU("agents on CPU, after learning", agCPU, 1);
#endif
	r->iBest = lastWinner;
	return r;
}


#pragma mark -
#pragma mark GPU - Only

// return a random number from 0 to nn-1
//__device__ __host__ unsigned rand_num(unsigned nn, unsigned *seeds, unsigned stride)
//{
//	unsigned r = RandUniformui(seeds, stride) % nn;
//	return r;
//}

// count the number of pieces on a board, storing the total in s_count[0]
// idx ranges from 0 to at least dc_half_board_size
__device__ void count_board_pieces(unsigned idx, unsigned *s_board, unsigned *s_count)
{
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
//		s_temp is a temporary area of size board_size/2
//		idx is the thread number which represents the cell, must range from 0 to (dc_board_size - 1)
//		seeds points to 1st seed for this agent
//		stride is the stride of seeds
__device__ void random_board(unsigned *s_board, unsigned *s_temp, unsigned *seeds, unsigned stride)
{
	unsigned idx = threadIdx.x;
	
	// randomly add pieces to cells using probability dc_piece_ratio
	if (idx < dc_board_size) {
		if (dc_piece_ratioX > RandUniform(seeds + idx, stride)) s_board[idx] = 1;
		else s_board[idx] = 0;
	}
	__syncthreads();
	
	// count the number of 1's
//	unsigned half = dc_half_board_size;
//	if (idx < half) s_temp[idx] = s_board[idx];
//	if (idx + half < dc_board_size) s_temp[idx] += s_board[idx + half];
//	__syncthreads();
//	while (0 < (half >>= 1)) {
//		if (idx < half) {
//			s_temp[idx] += s_temp[idx + half];
//		}
//		__syncthreads();
//	}
	count_board_pieces(idx, s_board, s_temp);
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


__device__ void random_board_masked(unsigned *s_board, unsigned *s_mask, unsigned *s_temp, unsigned *seeds, unsigned stride)
{
	unsigned idx = threadIdx.x;
	
	if (idx < dc_board_size) {
		if (dc_piece_ratioO > RandUniform(seeds + idx, stride) && !s_mask[idx]) s_board[idx] = 1;
		else s_board[idx] = 0;
	}
//	if (idx < dc_board_size && dc_piece_ratioO > RandUniform(seeds + idx, stride) && !s_mask[idx])
//		s_board[idx] = 1;
//	else s_board[idx] = 0;
	__syncthreads();
	
	// count the number of ones
//	unsigned half = dc_half_board_size;
//	if (idx < half) s_temp[idx] = s_board[idx];
//	if (idx + half < dc_board_size) s_temp[idx] += s_board[idx + half];
//	__syncthreads();
//	while (0 < (half >>= 1)) {
//		if (idx < half) s_temp[idx] += s_temp[idx + half];
//		__syncthreads();
//	}
	count_board_pieces(idx, s_board, s_temp);
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

__device__ void random_stateGPU(unsigned *s_state, float *s_temp, unsigned *s_seeds, unsigned stride)
{
	random_board(s_state, (unsigned *)s_temp, s_seeds, dc_board_size);
	random_board_masked(s_state + dc_board_size, s_state, (unsigned *)s_temp, s_seeds, dc_board_size);
}


// one thread block per state with at least board_size threads
//__global__ void randstate_kernel(unsigned *state, unsigned *seeds)
//{
//	unsigned idx = threadIdx.x;
//	unsigned iAgent = blockIdx.x * dc_state_size;
//	
//	__shared__ unsigned s_state[MAX_STATE_SIZE];
//	__shared__ unsigned s_temp[MAX_BOARD_SIZE];
//	
//	random_state(s_state, s_temp, idx, seeds + iAgent * 2, dc_board_size);
//	if (idx < dc_state_size) 
//		state[iAgent + idx] = s_state[idx];
//	if (blockDim.x < dc_state_size && (idx + blockDim.x) < dc_state_size)
//		state[iAgent + idx + blockDim.x] = s_state[idx + blockDim.x];
//}

// thread x dimension is 0 to board_size-1
// block x dimension is the board number
//__global__ void randboard_kernel(unsigned *board, unsigned *seeds)
//{
//	unsigned idx = threadIdx.x;
//	unsigned iAgent = blockIdx.x * dc_board_size;		// agent number times board_size
//	
//	__shared__ unsigned s_board[MAX_BOARD_SIZE];
//	__shared__ unsigned s_temp[MAX_BOARD_SIZE];
//	random_board(s_board, s_temp, idx, seeds + iAgent * 4, dc_board_size);
//
//	// copy the result to global memory
//	if(idx < dc_board_size) board[iAgent + idx] = s_board[idx];
//}

__device__ void rewardGPU(unsigned *s_state, unsigned *s_temp, unsigned *ps_terminal, float *ps_reward)
{
	unsigned idx = threadIdx.x;
	if (idx == 0){ *ps_terminal = 0; *ps_reward = 0.0f; }
	__syncthreads();
	
	count_board_pieces(idx, O_BOARDGPU(s_state), s_temp);
	if (idx == 0 && 0 == s_temp[0]) {
		*ps_terminal = 1;
		*ps_reward = REWARD_WIN;
	}
	__syncthreads();
	count_board_pieces(idx, X_BOARDGPU(s_state), s_temp);
	if (idx == 0 && 0 == s_temp[0]) {
		*ps_terminal = 1;
		*ps_reward = REWARD_LOSS;
	}
//	if (idx == 0) *ps_reward = 92.0f;
	__syncthreads();
}

// Calcualte the value for a state s using the specified weights,
// storing hidden activation in the specified location and stroing output value in *ps_V
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
	
	for (int iH = 0; iH < dc_num_hidden; iH++) {
		s_temp[idx] = s_hidden[iH] * (1.0f - s_hidden[iH]) * S_HO(s_wgts)[iH] * s_out[0] * (1.0f - s_out[0]);	// g_prime_j
		if (idx == 0) S_BH(s_e)[iH] += -1.0f * s_temp[idx];
		if (X_BOARDGPU(s_state)[idx]) S_IXH(s_e, iH)[idx] += s_temp[idx];
		if (O_BOARDGPU(s_state)[idx]) S_IOH(s_e, iH)[idx] += s_temp[idx];
	}
	__syncthreads();
}

__device__ void switch_sidesGPU(unsigned *s_state)
{
	unsigned idx = threadIdx.x;
	unsigned temp = s_state[idx];
	s_state[idx] = s_state[idx + dc_board_size];
	s_state[idx + dc_board_size] = temp;
}

__device__ void choose_moveGPU(unsigned *s_state, float *s_temp, float *s_wgts, float *s_hidden, float *s_out, unsigned *ps_terminal, float *ps_V)
{
	unsigned idx = threadIdx.x;
	
	// first see if the current board is terminal state
//	unsigned terminal;
	unsigned noVal = 1;
	float bestVal;
	unsigned iBestFrom;
	unsigned iBestTo;
//	float reward;

	// check for terminal condition
	rewardGPU(s_state, (unsigned *)s_temp, ps_terminal, ps_V);
	if (*ps_terminal) return;
	
	for (int iFrom = 0; iFrom < dc_board_size; iFrom++) {
		if (X_BOARDGPU(s_state)[iFrom]) {
			for (int m = 0; m < MAX_MOVES; m++) {
				int iTo = dc_moves[m * dc_board_size + iFrom];
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
	
	val_for_stateGPU(s_wgts, s_state, s_hidden, s_out, s_temp, ps_terminal, ps_V);
}

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

// copy weights from global memory to compact, shared memory
// idx is [0, board_size)
// g_wgts points to weights for this agent
__device__ void copy_wgts_to_s(float *g_wgts, float *s_wgts)
{
	unsigned idx = threadIdx.x;
	for (int iH = 0; iH < dc_num_hidden; iH++) {
		S_IXH(s_wgts, iH)[idx] = G_IXH(g_wgts, iH)[idx];
		S_IOH(s_wgts, iH)[idx] = G_IOH(g_wgts, iH)[idx];
	}
	
	if (idx < dc_num_hidden) {
		S_BH(s_wgts)[idx] = G_BH(g_wgts)[idx];
		S_HO(s_wgts)[idx] = G_HO(g_wgts)[idx];
	}

	if (idx == 0) {
		S_BO(s_wgts)[idx] = G_BO(g_wgts)[idx];
	}
	__syncthreads();
}

__device__ void copy_wgts_to_g(float *s_wgts, float *g_wgts)
{
	unsigned idx = threadIdx.x;
	for (int iH = 0; iH < dc_num_hidden; iH++) {
		G_IXH(g_wgts, iH)[idx] = S_IXH(s_wgts, iH)[idx];
		G_IOH(g_wgts, iH)[idx] = S_IOH(s_wgts, iH)[idx];
	}
	
	if (idx < dc_num_hidden) {
		G_BH(g_wgts)[idx] = S_BH(s_wgts)[idx];
		G_HO(g_wgts)[idx] = S_HO(s_wgts)[idx];
	}

	if (idx == 0) {
		G_BO(g_wgts)[idx] = S_BO(s_wgts)[idx];
	}
	__syncthreads();
}


// block number is the agent number,
// threads per block is board_size
__global__ void learn_kernel(unsigned *seeds, float *wgts, float *e, float *ag2_wgts)
{
	unsigned idx = threadIdx.x;
	unsigned iAgent = blockIdx.x;
//	unsigned iAgentBS = blockIdx.x * dc_board_size;
//	unsigned iAgentWgts = blockIdx.x * dc_wgts_stride;
	
	// static shared memory
	__shared__ float s_rand;
	__shared__ float s_reward;
	__shared__ float s_lambda;
	__shared__ float s_alpha;
	__shared__ float s_V;
	__shared__ float s_V_prime;
	__shared__ float s_delta;
	__shared__ unsigned s_terminal;
	__shared__ unsigned s_games;
	__shared__ unsigned s_wins;
	__shared__ unsigned s_losses;
	__shared__ unsigned s_tempui;
	__shared__ float s_tempf;

	// dynamic shared memory
	extern __shared__ unsigned s_seeds[];						// 4 * dc_board_size
	unsigned *s_state = s_seeds + 4 * dc_board_size;			// dc_state_size
	float *s_temp = (float *)(s_state + dc_state_size);			// dc_board_size
	float *s_hidden = s_temp + dc_board_size;		// dc_num_hidden
	float *s_out = s_hidden + dc_num_hidden;					// dc_num_hidden
	float *s_ophidden = s_out + dc_num_hidden;					// dc_num_hidden
	float *s_wgts = s_ophidden + dc_num_hidden;					// dc_num_wgts
	float *s_e = s_wgts + dc_num_wgts;							// dc_num_wgts
	float *s_opwgts = s_e + dc_num_wgts;						// dc_num_wgts
	
	// copy values to shared memory
	if (idx == 0){
		s_lambda = dc_ag.lambda[iAgent];
		s_alpha = dc_ag.alpha[iAgent];
		s_games = 0;
		s_wins = 0;
		s_losses = 0;
	}
	__syncthreads();
	
	s_state[idx] = dc_ag.states[iAgent * dc_state_size + idx];
	s_state[idx + dc_board_size] = dc_ag.states[iAgent * dc_state_size +idx + dc_board_size];
	
	s_seeds[idx] = seeds[iAgent * 4 * dc_board_size + idx];
	s_seeds[idx + dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + dc_board_size];
	s_seeds[idx + 2*dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + 2*dc_board_size];
	s_seeds[idx + 3*dc_board_size] = seeds[iAgent * 4 * dc_board_size + idx + 3*dc_board_size];
	
	copy_wgts_to_s(wgts + iAgent * dc_wgts_stride, s_wgts);
	copy_wgts_to_s(e + iAgent * dc_wgts_stride, s_e);
	copy_wgts_to_s(ag2_wgts + iAgent * dc_wgts_stride, s_opwgts);

	unsigned turn = 0;
	unsigned total_turns = 0;
	
	// skip this for testing so agent starts with the same initial state as CPU
//	random_stateGPU(s_state, s_temp, s_seeds, dc_board_size);

//	if (idx == 0) s_rand = RandUniform(s_seeds, dc_board_size);
//	__syncthreads();

	
	s_reward = 0.0f;
	if (dc_ag.next_to_play[iAgent]) {
		take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward);
		++turn;
	}
	__syncthreads();
	
	// exit 0

	choose_moveGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, &s_V);
	
	update_traceGPU(s_state, s_wgts, s_e, s_hidden, s_out, s_lambda, s_temp);
	
	// exit 1
	
	while (total_turns++ < dc_episode_length) {
		take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_terminal, s_ophidden, &s_reward);
		++turn;
		
//		break;	// exit 2
		
		if (s_terminal || (turn == dc_max_turns)) {

//			break;	// exit 5

			turn = 0;
			if (idx == 0) {
				++s_games;
				if (s_terminal){
					if (s_reward > 0.50f) ++s_wins;
					else ++s_losses;
				}
				s_rand = RandUniform(s_seeds, dc_board_size);
			}
			__syncthreads();			
			random_stateGPU(s_state, s_temp, s_seeds, dc_board_size);

//			break;	// exit 6

			if (s_rand < 0.50f) {
				take_actionGPU(s_state, s_temp, s_opwgts, s_hidden, s_out, &s_tempui, s_ophidden, &s_tempf);
				++turn;
			}

//			break;	// exit 7

		}
		
		choose_moveGPU(s_state, s_temp, s_wgts, s_hidden, s_out, &s_tempui, &s_V_prime);

//		if (s_games > 0) break;	// exit 8

		if (idx == 0){
			s_delta = s_reward + (s_terminal ? 0.0f : (dc_gamma * s_V_prime)) - s_V;
		}
		__syncthreads();
		
//		dc_ag.epsilon[iAgent] = s_reward;	// stach the delta value in agent's epsilon for debugging
//		break;	// exit 3

//		if (s_games > 0){
//			dc_ag.epsilon[iAgent] = s_delta;
//			break;					// exit 9
//		}

		update_wgtsGPU(s_alpha, s_delta, s_wgts, s_e);
		if (s_terminal) reset_traceGPU(s_e);
		update_traceGPU(s_state, s_wgts, s_e, s_hidden, s_out, s_lambda, s_temp);

//		dc_ag.epsilon[iAgent] = s_reward;	// stach the delta value in agent's epsilon for debugging
//		break;	// exit 4
		
//		if (s_games > 0){
//			dc_ag.epsilon[iAgent] = s_delta;
//			break;					// exit 10
//		}

		if (idx == 0) s_V = s_V_prime;
		__syncthreads();
	}
	
	// copy values back to global memory
	dc_ag.states[iAgent * dc_state_size + idx] = s_state[idx];
	dc_ag.states[iAgent * dc_state_size + idx + dc_board_size] = s_state[idx + dc_board_size];
	
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx] = s_seeds[idx];
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx + dc_board_size] = s_seeds[idx + dc_board_size];
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx + 2*dc_board_size] = s_seeds[idx + 2*dc_board_size];
	dc_ag.seeds[iAgent * dc_board_size * 4 + idx + 3*dc_board_size] = s_seeds[idx + 3*dc_board_size];

	copy_wgts_to_g(s_wgts, wgts + iAgent * dc_wgts_stride);
	copy_wgts_to_g(s_e, e + iAgent * dc_wgts_stride);

}

// calculate the 
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
	printf("%d elements in shared memory, total of %d bytes\n", count, count * 4);
	return sizeof(unsigned) * count;
}

RESULTS *runGPU(AGENT *agGPU, float *champ_wgts)
{
	printf("running on GPU...\n");
#ifdef DUMP_INITIAL_AGENTS
	dump_agentsGPU("initial agents on GPU", agGPU, 1);
#endif
	dim3 blockDim(g_p.board_size);
	dim3 gridDim(g_p.num_agents);

	PRE_KERNEL("learn_kernel");
	learn_kernel<<<gridDim, blockDim, dynamic_shared_mem()>>>(agGPU->seeds, agGPU->wgts, agGPU->e, agGPU->wgts);
	POST_KERNEL(learn_kernel);
#ifdef DUMP_FINAL_AGENTS_GPU
	dump_agentsGPU("agents after learning on GPU", agGPU, 1);
#endif
	return NULL;
}




