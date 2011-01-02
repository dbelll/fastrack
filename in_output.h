/*
 *  in_output.h
 *  fastrack
 *
 *  Created by Dwight Bell on 1/1/11.
 *  Copyright 2011 dbelll. All rights reserved.
 *
 */

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
	fprintf(f, "NUM_AGENTS, %d, OP_FRACTION %d\n", g_p.num_agents, g_p.op_fraction);
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

void dump_agentsGPU(const char *str, AGENT *agGPU, unsigned dumpW, unsigned dumpSaved)
{
	printf("dump_agentsGPU\n");
	
	// create a CPU copy of the GPU agent data
	AGENT *agCPU = (AGENT *)malloc(sizeof(AGENT));
	
	agCPU->seeds = host_copyui(agGPU->seeds, 4 * g_p.num_agents * g_p.board_size);
	agCPU->states = host_copyui(agGPU->states, g_p.num_agents * g_p.state_size);
	agCPU->next_to_play = host_copyui(agGPU->next_to_play, g_p.num_agents);
	agCPU->wgts = host_copyf(agGPU->wgts, g_p.num_agent_floats * g_p.num_agents);
	set_agent_float_pointers(agCPU);
	
	dump_agentsCPU(str, agCPU, dumpW, dumpSaved);
	
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

void dump_agent(AGENT *agCPU, unsigned iag, unsigned dumpW, unsigned dumpSaved)
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
	
	if (dumpSaved) {
		dump_wgts_header("[savedwgts]");
		float *pSaved = agCPU->saved_wgts + iag * g_p.wgts_stride;
		dump_wgts_BH(pSaved);
		for (int i = 0; i < g_p.state_size; i++) dump_wgts_IH(pSaved, i);
		dump_wgts_HO(pSaved);
		dump_wgts_BO(pSaved);
	}
	
	printf("[   alpha], %9.4f\n", agCPU->alpha[iag]);
	printf("[ epsilon], %9.4f\n", agCPU->epsilon[iag]);
	printf("[  lambda], %9.4f\n", agCPU->lambda[iag]);
#endif	
	dump_state(agCPU->states + iag*g_p.state_size, 0, agCPU->next_to_play[iag]);
}

// dump all agents, flag controls if the eligibility trace values are also dumped
void dump_agentsCPU(const char *str, AGENT *agCPU, unsigned dumpW, unsigned dumpSaved)
{
	printf("======================================================================\n");
	printf("%s\n", str);
	printf("----------------------------------------------------------------------\n");
	for (int i = 0; i < g_p.num_agents; i++) {
		printf("\n[AGENT%5d] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n", i);
		dump_agent(agCPU, i, dumpW, dumpSaved);
	}
	printf("======================================================================\n");
	
}

