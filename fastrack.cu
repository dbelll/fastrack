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
__constant__ unsigned dc_num_wgts;
__constant__ float dc_piece_ratioX;	// ratio of num_pieces to board_size
__constant__ float dc_piece_ratioO;	// ratio of num_pieces to (board_size - num_pieces)

#pragma mark -
#pragma mark Helpers

//	Initialize the global seeds using specified seed value.
void set_global_seeds(unsigned seed)
{
	srand(seed);
	for (int i = 0; i < 4; i++) {
		g_seeds[i] = rand();
	}
	printf("global seeds are: %u %u %u %u\n", g_seeds[0], g_seeds[1], g_seeds[2], g_seeds[3]);
}

float sigmoid(float x)
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

// generate a random state with n pieces for ech player
void random_state(unsigned *state, unsigned n, unsigned *seeds, unsigned stride)
{
	random_board(X_BOARD(state), n, seeds, stride);
	random_board_masked(O_BOARD(state), X_BOARD(state), n, seeds, stride);
}



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
		printf("   %4d games vs Champ\n", CHAMP_GAMES);

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
	static unsigned version = 1;
	
	fprintf(f, "%d\n", version);
	fprintf(f, "%d\n", g_p.board_width);
	fprintf(f, "%d\n", g_p.board_height);
	fprintf(f, "%d\n", g_p.num_pieces);
	fprintf(f, "%d\n", g_p.max_turns);
	fprintf(f, "%d\n", g_p.num_hidden);
	fprintf(f, "%d\n", g_p.num_wgts);
	for (int i = 0; i < g_p.num_wgts; i++) {
		fprintf(f, "%f\n", ag->wgts[iAg * g_p.num_wgts + i]);
	}
	fclose(f);
}

void read_agent(FILE *f, AGENT *ag, unsigned iAg)
{
	unsigned version;
	unsigned board_width, board_height;
	unsigned num_pieces;
	unsigned max_turns;
	unsigned num_hidden;
	unsigned num_wgts;
	
	fscanf(f, "%d", &version);

	switch (version) {
		case 1:
			fscanf(f, "%u", &board_width);
			fscanf(f, "%u", &board_height);
			if (board_width != g_p.board_width || board_height != g_p.board_height) {
				printf("*** AGENT ERROR *** board size mismatch\n");
				break;
			}
			fscanf(f, "%u", &num_pieces);
			fscanf(f, "%u", &max_turns);
			fscanf(f, "%u", &num_hidden);
			fscanf(f, "%u", &num_wgts);
			for (int i = 0; i < num_wgts; i++) {
				fscanf(f, "%f", ag->wgts + iAg * g_p.num_wgts + i);
			}
			break;
		default:
			break;
	}
}

unsigned read_num_wgts(FILE *f)
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
			break;
		default:
			break;
	}
	return num_wgts;
}

unsigned read_num_hidden(FILE *f)
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
			break;
		default:
			break;
	}
	printf("read_num_hidden: (%dx%d) board, %d pieces, %d turns, %d num_hidden, %d num_wgts\n", board_width, board_height, num_pieces, max_turns, num_hidden, num_wgts);
	return num_hidden;
}

// read in just the weights from an agent file
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
			for (int i = 0; i < num_wgts; i++) {
				fscanf(f, "%f", wgts + i);
//				printf("wgt %d is %9.6f\n", i, wgts[i]);
			}
			break;
		default:
			break;
	}
	printf("read_wgts: (%dx%d) board, %d pieces, %d turns, %d num_hidden, %d num_wgts\n", board_width, board_height, num_pieces, max_turns, num_hidden, num_wgts);
}

// load weights for champ from a file
float *load_champ(const char *file)
{
	FILE *f = fopen(file, "r");
	if (!f) {
		printf("could not open champ file %s\n", file);
		exit(-1);
	}
	
	// allocate an array of the appropriate size to hold champ's weights
	unsigned num_wgts = read_num_wgts(f);
	float *wgts = (float *)malloc(num_wgts * sizeof(float));
	
	// now read the weights from the file to fill the array
	read_wgts(f, wgts);
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

// straight dump of weight values
void dump_wgts(float *wgts)
{
	for (int i = 0; i < g_p.num_hidden; i++) {
		printf(", %9.4f", wgts[i]);
	}
	printf("\n");
}

// print out all weights in a formatted output
void dump_all_wgts(float *wgts, unsigned num_hidden)
{
	dump_wgts_header("[ WEIGHTS]");
	// get the weight pointer for this agent
	printf("[    B->H]"); dump_wgts(wgts);
	for (int i = 0; i < g_p.state_size; i++){
		printf("[IN%03d->H]", i); dump_wgts(wgts + (1+i) * num_hidden);
	}
	printf("[    H->O]"); dump_wgts(wgts + (1+g_p.state_size) * num_hidden);
	printf("[    B->O], %9.4f\n\n", wgts[(2+g_p.state_size) * num_hidden]);
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

// dump all agents, flag controls if the eligibility trace values are also dumped
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
// calculates agent pointers based on offset from ag->wgts
// agent data is organized as follows
void set_agent_float_pointers(AGENT *ag)
{
	ag->e = ag->wgts + g_p.num_wgts * g_p.num_agents;
	ag->saved_wgts = ag->e + g_p.num_wgts * g_p.num_agents;
	ag->alpha = ag->saved_wgts + g_p.num_wgts * g_p.num_agents;
	ag->epsilon = ag->alpha + g_p.num_agents;
	ag->lambda = ag->epsilon + g_p.num_agents;
}

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

// Reset all agent weights to random initial values and reset trace to 0.0f
void randomize_agent(AGENT *ag)
{
	for (int i=0; i < g_p.num_wgts * g_p.num_agents; i++){
		ag->wgts[i] = g_p.init_wgt_min + (g_p.init_wgt_max - g_p.init_wgt_min) * RandUniform(g_seeds, 1);
		ag->e[i] = 0.0f;
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

	// save the parameters and calculate any global constants based on the parameters
	g_p = p;
	set_global_seeds(p.seed);
	build_move_array();
	build_start_state();

	// allocate and initialize the agent data on CPU
	AGENT *ag = (AGENT *)malloc(sizeof(AGENT));

	ag->seeds = (unsigned *)malloc(4 * p.num_agents * p.board_size * sizeof(unsigned));
	for (int i = 0; i < 4*p.num_agents * p.board_size; i++) ag->seeds[i] = rand();
	
	ag->wgts = (float *)malloc(p.num_agent_floats * p.num_agents * sizeof(float));
	set_agent_float_pointers(ag);	
	randomize_agent(ag);
	
	for (int i = 0; i < p.num_agents; i++) { 
		set_agent_params(ag, i, p.alpha, p.epsilon, p.lambda); 
	}
	
	return ag;
}


AGENT *init_agentsGPU(AGENT *agCPU)
{
	AGENT *agGPU = (AGENT *)malloc(sizeof(AGENT));
	agGPU->seeds = device_copyui(agCPU->seeds, 4 * g_p.num_agents * g_p.board_size);
	agGPU->wgts = device_copyf(agCPU->wgts, g_p.num_agent_floats * g_p.num_agents);
	set_agent_float_pointers(agGPU);
	
	cudaMemcpyToSymbol("dc_ag", &agGPU, sizeof(AGENT));
	cudaMemcpyToSymbol("dc_gamma", &g_p.gamma, sizeof(float));
	cudaMemcpyToSymbol("dc_num_hidden", &g_p.num_hidden, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_num_pieces", &g_p.num_pieces, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_state_size", &g_p.state_size, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_board_size", &g_p.board_size, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_num_wgts", &g_p.num_wgts, sizeof(unsigned));
	
	unsigned half_board_size = 1;
	while (half_board_size < g_p.board_size) {
		half_board_size <<= 1;
	}
	half_board_size >>=1;	// now half_board_size is the largest power of two less board_size
	cudaMemcpyToSymbol("dc_half_board_size", &half_board_size, sizeof(unsigned));
	
	float piece_ratioX = (float)g_p.num_pieces / (float)g_p.board_size;
	float piece_ratioO = (float)g_p.num_pieces / (float)(g_p.board_size - g_p.num_pieces);
	cudaMemcpyToSymbol("dc_piece_ratioX", &piece_ratioX, sizeof(float));
	cudaMemcpyToSymbol("dc_piece_ratioO", &piece_ratioO, sizeof(float));
	
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

float take_action2(unsigned *state, float *owgts, float *hidden, float *out, unsigned *terminal, unsigned ohidden)
{
//	printf("on entry g_p.num_hidden = %d, ohidden = %d\n", g_p.num_hidden, ohidden);

	float r = reward(state, terminal);
	if (*terminal) return r;	// given state is terminal, just return the reward

	// set param consistent with opponent
	unsigned save_hidden = g_p.num_hidden;
	unsigned save_num_wgts = g_p.num_wgts;
	g_p.num_hidden = ohidden;
	g_p.num_wgts = g_p.num_hidden * (2 * g_p.board_size + 3);
	
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
		(ag2_wgts	? take_action2(state, ag2_wgts, hidden, out, &terminal, ag2_hidden)
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
		float reward = (ag2_wgts	? take_action2(state, ag2_wgts, hidden, out, &terminal, ag2_hidden)
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
WON_LOSS auto_learn(AGENT *ag1, unsigned iAg, float *ag2_wgts, unsigned start_pieces, unsigned num_turns, unsigned max_turns, unsigned ag2_hidden, unsigned *seeds, unsigned stride)
{
//	printf("auto_learning: %d pieces,  %d turns, with %d max turns per game...\n", start_pieces, num_turns, max_turns);

	float *ag1_wgts = ag1->wgts + iAg * g_p.num_wgts;	// points to start of learning agent's weights
	float *ag1_e = ag1->e + iAg * g_p.num_wgts;			// points to start of learning agent's trace
	
	if (!ag1) {
		printf("***ERROR *** random agent can not learn!!!\n");
		exit(-1);
	}
	
	WON_LOSS wl = {0, 0, 0, 0};
	
#ifdef DUMP_ALL_AGENT_UPDATES
	dump_agent(ag1, iAg, 1);
#endif
	
	static unsigned *state = NULL;
	static float *hidden = NULL;
	static float *out = NULL;
	if(!state) state = (unsigned *)malloc(g_p.state_size * sizeof(unsigned));
	if(!hidden) hidden = (float *)malloc(g_p.num_hidden * sizeof(float));
	if(!out) out = (float *)malloc(g_p.num_hidden * sizeof(float));
	
	unsigned turn = 0;
	unsigned total_turns = 0;
	unsigned terminal = 0;
	
	// set up the starting state
	set_start_state(state, start_pieces, seeds, stride);
	if (RandUniform(seeds, stride) < 0.50f) {
#ifdef DUMP_MOVES
		printf("New game, O to play first...\n");
		dump_state(state, turn, 1);		
#endif
		float r = (ag2_wgts	? take_action2(state, ag2_wgts, hidden, out, &terminal, ag2_hidden) 
							: take_random_action(state, &terminal, seeds, stride));
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
	float V = choose_move(state, ag1_wgts, hidden, out);
	
	update_trace(state, ag1_wgts, ag1_e, hidden, out, ag1->lambda[iAg]);

#ifdef DUMP_ALL_AGENT_UPDATES
	printf("after updating trace...\n");
	dump_agent(ag1, iAg, 1);
#endif
	
	// loop over the number of turns
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

		

		float reward = (ag2_wgts	? take_action2(state, ag2_wgts, hidden, out, &terminal, ag2_hidden) 
									: take_random_action(state, &terminal, seeds, stride));
		++turn;
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
		if (terminal || (turn == max_turns)) {
#ifdef DUMP_MOVES
			if (!terminal) printf("****** GAME OVER: reached maximum number of turns per game (total_turns = %d, games = %d)\n", total_turns, wl.games);
#endif
			++wl.games;
//			if (++wl.games < num_turns) {
#ifdef DUMP_MOVES
				printf("\n\n--------------- game %d ---------------------\n", wl.games);
#endif
				turn = 0;
				set_start_state(state, start_pieces, seeds, stride);
				if (RandUniform(seeds, stride) < 0.50f) {
#ifdef DUMP_MOVES
					printf("New game, O to play first...\n");
					dump_state(state, turn, 1);		
#endif
					(ag2_wgts	? take_action(state, ag2_wgts, hidden, out, &terminal) 
								: take_random_action(state, &terminal, seeds, stride));
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
		}
//		printf("choosing next move...\n");
		float V_prime = choose_move(state, ag1_wgts, hidden, out);
		float delta = reward + (terminal ? 0.0f : (g_p.gamma * V_prime)) - V;
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

#ifdef DUMP_ALL_AGENT_UPDATES
		printf("after updating trace:\n");
		dump_agent(ag1, iAg, 1);
#endif
		
		V = V_prime;
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
	
	// allocate the structure to hold the results
	RESULTS *r = newResults(g_p);
	
//	FILE *champFile = fopen(AGENT_FILE_CHAMP, "r");
//	unsigned champ_num_hidden = read_num_hidden(champFile);
////	printf("champ from file %s has %d hidden nodes\n", AGENT_FILE_CHAMP, champ_num_hidden);
//	float *champ_wgts = (float *)malloc(champ_num_hidden * (2*g_p.board_size + 3) * sizeof(float));
//	read_wgts(champFile, champ_wgts);
//	fclose(champFile);
	
	// each episode, the agent's weights are saved to be used as opponents
//	float *saved_wgts = (float *)malloc(g_p.num_agents * g_p.num_wgts * sizeof(float));
	
	// standings holds the won/loss record during learning
//	WON_LOSS *standings = (WON_LOSS *)malloc(g_p.num_agents * sizeof(WON_LOSS));
	
	// vsChamp hods won-loss record when each agent is run against the benchmark after learning
//	WON_LOSS *vsChamp = (WON_LOSS *)malloc(g_p.num_agents * sizeof(WON_LOSS));
	
	// lastWinner is the agent in first place after each learning session
	unsigned lastWinner = 0;


//	printf("warm-up versus RAND\n");
//	dump_agentsCPU("prior to warmup", agCPU, 0);
	for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
		unsigned save_num_pieces = g_p.num_pieces;
		printf("agent %d learning against RAND ... ", iAg);
		for (int p = 1; p <= save_num_pieces; p++) {
			g_p.num_pieces = p;
			printf(" %d ... ", p);
			auto_learn(agCPU, iAg, NULL, g_p.num_pieces, g_p.warmup_length, g_p.max_turns, 0, agCPU->seeds + iAg, g_p.num_agents);
		}
		printf(" done\n");
		g_p.num_pieces = save_num_pieces;
	}

	for (int iSession = 0; iSession < g_p.num_sessions; iSession++) {
		// copy the current weights to the saved_wgts area
		memcpy(agCPU->saved_wgts, agCPU->wgts, g_p.num_agents * g_p.num_wgts * sizeof(float));
		
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
			for (int iOp = 0; iOp < g_p.num_agents/((iSession > 0) ? 2 : 1); iOp++) {
				unsigned xOp = (iAg + iOp) % g_p.num_agents;
				if (iSession > 0) xOp = r->standings[(iSession-1) * g_p.num_agents + iOp].agent;
//				printf("(%d vs %d) ", iAg, xOp);
				WON_LOSS wl = auto_learn(agCPU, iAg, agCPU->saved_wgts + xOp * g_p.num_wgts, g_p.num_pieces, g_p.episode_length, g_p.max_turns, g_p.num_hidden, agCPU->seeds + iAg, g_p.num_agents);
//				printf("g_p.num_hidden is %d\n", g_p.num_hidden);
				r->standings[iStand].games += wl.games;
				r->standings[iStand].wins += wl.wins;
				r->standings[iStand].losses += wl.losses;
			}
			
			// compete against the champ as a benchmark
			r->vsChamp[iStand] = compete(agCPU->wgts + iAg * g_p.num_wgts, NULL, champ_wgts, NULL, g_p.num_pieces, CHAMP_GAMES, g_p.max_turns, 0, g_p.num_hidden, agCPU->seeds + iAg, g_p.num_agents);
			
			// write results to file
//			fprintf(f, "%d, %d, %d, %d, %d, %d, %d, %d\n", iSession, iAg, r->standings[iStand].games, r->standings[iStand].wins, r->standings[iStand].losses, r->vsChamp[iStand].games, r->vsChamp[iStand].wins, r->vsChamp[iStand].losses);
			
		}
		
		// sort and print the standings
		print_standings(r->standings + iSession * g_p.num_agents, r->vsChamp + iSession * g_p.num_agents);
		
		// remember the winner after this session
		lastWinner = r->standings[iSession * g_p.num_agents + 0].agent;
	}
	
//	// write the best agent to the AGENT_FILE_OUT
//	FILE *agfile = fopen(AGENT_FILE_OUT, "w");
//	save_agent(agfile, agCPU, lastWinner);
//	fclose(agfile);

	// as a final test, see if the last winner can beat the champ over 1000 games
	WON_LOSS wlChamp = compete(agCPU->wgts + lastWinner * g_p.num_wgts, "CHALLENGER", champ_wgts, "CHAMP", g_p.num_pieces, 1000, g_p.max_turns, 0, g_p.num_hidden, agCPU->seeds + lastWinner, g_p.num_agents);
	printf("CHALLENGER v CHAMP  G: %d  W: %d  L: %d    %+4d   ", wlChamp.games, wlChamp.wins, wlChamp.losses, wlChamp.wins - wlChamp.losses);

	if (wlChamp.wins < wlChamp.losses) {
		printf("CHAMP wins\n");
	}else {
		printf("CHALLENGER beat CHAMP!!!\n");
	}

	// show a few games of the best against itself
#ifdef SHOW_SAMPLE_GAMES_AFTER_LEARNING
	compete(agCPU->wgts + lastWinner * g_p.num_wgts, "BEST", agCPU->wgts + lastWinner * g_p.num_wgts, "BEST", g_p.num_pieces, SHOW_SAMPLE_GAMES_AFTER_LEARNING, g_p.max_turns, 1);
#endif
//	fclose(f);
//	free(standings);
//	free(champ_wgts);
	
	r->iBest = lastWinner;
	return r;
}


#pragma mark -
#pragma mark GPU - Only

// return a random number from 0 to n-1
__device__ unsigned rand_num(unsigned nn, unsigned *seeds, unsigned stride)
{
//	unsigned r = n;
//	
//	// mask1 masks off the bits that are needed for a number from [0 to n-1]
//	unsigned mask = 1;
//	while (mask < n) {
//		mask <<= 1;
//	}
//	mask -= 1;
//	do {
//		r = RandUniformui(seeds, stride) & mask;
//	} while (r >= n);
	unsigned r = RandUniformui(seeds, stride) % nn;
	return r;
}

// Fill in the board with random pieces (uses dc_board_size and dc_num_pieces)
//		s_board points to the board for this agent
//		s_temp is a temporary area of size board_size/2
//		idx is the thread number which represents the cell
//		seeds points to 1st seed for this agent
//		stride is the stride of seeds
__device__ void random_board(unsigned *s_board, unsigned *s_temp, unsigned idx, unsigned *seeds, unsigned stride)
{
	// randomly add pieces to some cells using probability dc_piece_ratio
	if (idx < dc_board_size && dc_piece_ratioX > RandUniform(seeds + idx, stride))
			s_board[idx] = 1;
	else s_board[idx] = 0;

	__syncthreads();
	
	// count the number of 1's
	unsigned half = dc_half_board_size;
	if (idx < half) s_temp[idx] = s_board[idx];
	if (idx + half < dc_board_size) s_temp[idx] += s_board[idx + half];
	__syncthreads();
	while (0 < (half >>= 1)) {
		if (idx < half) {
			s_temp[idx] += s_temp[idx + half];
		}
		__syncthreads();
	}
	// s_temp[0] now contains the number of 1's
	
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


__device__ void random_board_masked(unsigned *s_board, unsigned *s_mask, unsigned *s_temp, unsigned idx, unsigned *seeds, unsigned stride)
{
	if (idx < dc_board_size && dc_piece_ratioO > RandUniform(seeds + idx, stride) && !s_mask[idx])
		s_board[idx] = 1;
	else s_board[idx] = 0;
	__syncthreads();
	
	// count the number of ones
	unsigned half = dc_half_board_size;
	if (idx < half) s_temp[idx] = s_board[idx];
	if (idx + half < dc_board_size) s_temp[idx] += s_board[idx + half];
	__syncthreads();
	while (0 < (half >>= 1)) {
		if (idx < half) s_temp[idx] += s_temp[idx + half];
		__syncthreads();
	}
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

__device__ void random_state(unsigned *s_state, unsigned *s_temp, unsigned idx, unsigned *seeds, unsigned stride)
{
	random_board(s_state, s_temp, idx, seeds + idx, dc_board_size);
	random_board_masked(s_state + dc_board_size, s_state, s_temp, idx, seeds + idx, dc_board_size);
}


// one thread block per state with at least board_size threads
__global__ void randstate_kernel(unsigned *state, unsigned *seeds)
{
	unsigned idx = threadIdx.x;
	unsigned iAgent = blockIdx.x * dc_state_size;
	
	__shared__ unsigned s_state[MAX_STATE_SIZE];
	__shared__ unsigned s_temp[MAX_BOARD_SIZE];
	
	random_state(s_state, s_temp, idx, seeds + iAgent * 2, dc_board_size);
	if (idx < dc_state_size) 
		state[iAgent + idx] = s_state[idx];
	if (blockDim.x < dc_state_size && (idx + blockDim.x) < dc_state_size)
		state[iAgent + idx + blockDim.x] = s_state[idx + blockDim.x];
}

// thread x dimension is 0 to board_size-1
// block x dimension is the board number
__global__ void randboard_kernel(unsigned *board, unsigned *seeds)
{
	unsigned idx = threadIdx.x;
	unsigned iAgent = blockIdx.x * dc_board_size;		// agent number times board_size
	
	__shared__ unsigned s_board[MAX_BOARD_SIZE];
	__shared__ unsigned s_temp[MAX_BOARD_SIZE];
	random_board(s_board, s_temp, idx, seeds + iAgent * 4, dc_board_size);

	// copy the result to global memory
	if(idx < dc_board_size) board[iAgent + idx] = s_board[idx];
}

RESULTS *runGPU(AGENT *agGPU, float *champ_wgts)
{
	printf("running on GPU...\n");
	// generate a random board
	unsigned *d_state = device_allocui(g_p.state_size * g_p.num_agents);
	
	unsigned thx = 1;
	while (thx < g_p.state_size){
		 thx <<= 1;
	}
	printf("thx = %d\n", thx);
	dim3 blockDim(g_p.board_size);
	dim3 gridDim(g_p.num_agents);
	printf("bockDim is (%dx%d) and gridDim is (%dx%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	randstate_kernel<<<gridDim, blockDim>>>(d_state, agGPU->seeds);
	
	device_dumpui("raw state on device", d_state, 2 * g_p.board_height * g_p.num_agents, g_p.board_width);
	dump_statesGPU(d_state, g_p.num_agents);
	
	return NULL;
}




