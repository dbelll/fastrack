//
//  main.c
//  fastrack
//
//  Created by Dwight Bell on 12/13/10.
//  Copyright dbelll 2010. All rights reserved.
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "cuda_utils.h"
#include "./common/inc/cutil.h"
#include "fastrack.h"
#include "helpers.h"

// print out information on using this program
void display_help()
{
	printf("fastrack usage: fastrack [parameters+]\n");
	printf("fastrack parameters:\n");
	printf("  --SEED                a number from 0 to 15 for varying random numbers\n");
	printf("  --BOARD_SIZE			board size, encoded as 1000 * width + height\n");
	printf("  --NUM_PIECES			number of random pieces for each side, 0 ==> one row for each player");
	printf("  --MAX_TURNS			maximum turns per game");

	printf("  --NUM_HIDDEN          number of hidden nodes in agent nn");
	printf("  --INIT_THETA_MIN		minimum of range of possible initial weight values\n");
	printf("  --INIT_THETA_MAX		maximum of range of possible initial weight values\n");
	
	printf("  --NUM_AGENTS			number of agents\n");
	printf("  --NUM_SESSIONS        number of learning sessions, each session is round-robin");
	printf("  --EPISODE_LENGTH      number of turns for learning against each opponent in each episode\n");
	printf("  --WARMUP_LENGTH	    number of turns for initial learning vs. random-playing agent\n");
	printf("  --ALPHA               float value for alpha, the learning rate parameter\n");
	printf("  --EPSILON             float value for epsilon, the exploration parameter\n");
	printf("  --GAMMA               float value for gamma, the discount factor\n");
	printf("  --LAMBDA              float value for lambda, the trace decay factor\n");

	printf("  --CPU                 run on CPU");
	printf("  --GPU                 run on CPU");
	printf("  --HELP                print this help message\n");
	printf("default values will be used for any parameters not on command line\n");
}

void dump_params(PARAMS p)
{
	printf("game parameters:\n");
	printf("   seed          %10d\n", p.seed);
	printf("   board          (%3d x%3d)\n", p.board_width, p.board_height);
	printf("   num_pieces       %7d\n", p.num_pieces);
	printf("   max_turns		%7d\n", p.max_turns);
	printf("   board_size       %7d\n", p.board_size);
	printf("   state_size       %7d\n", p.state_size);
	printf("agent parameters:\n");
	printf("   num_hidden       %7d\n", p.num_hidden);
	printf("   num_wgts		    %7d\n", p.num_wgts);
	printf("   num_agent_floats %7d\n", p.num_agent_floats);
	printf("   init_wgt_min     %7.4f\n", p.init_wgt_min);
	printf("   init_wgt_max     %7.4f\n", p.init_wgt_max);
	printf("learning parameters:\n");
	printf("   num_agents       %7d\n", p.num_agents);
	printf("   num_sessions     %7d\n", p.num_sessions);
	printf("   episode_length   %7d\n", p.episode_length);
	printf("   warmup_length    %7d\n", p.warmup_length);
	printf("   alpha            %7.4f\n", p.alpha);
	printf("   epsilon          %7.4f\n", p.epsilon);
	printf("   gamma            %7.4f\n", p.gamma);
	printf("   lambda           %7.4f\n", p.lambda);
	printf("   run on CPU?      %s\n", p.run_on_CPU ? "TRUE" : "FALSE");
	printf("   run on GPU?      %s\n", p.run_on_GPU ? "TRUE" : "FALSE");
}

// read parameters from command line (or use default values) and print the header for this run
PARAMS read_params(int argc, const char **argv)
{
#ifdef VERBOSE
	printf("reading parameters...\n");
#endif
	
	PARAMS p;
	if (PARAM_PRESENT("HELP")) { display_help(); exit(0); }

	p.seed = GET_PARAM("SEED", 1000);
	unsigned bs = GET_PARAM("BOARD_SIZE", 4004);		// one integer = width * 1000 + height
	p.board_width = bs / 1000;
	p.board_height = bs % 1000;
	if (p.board_width > MAX_BOARD_DIMENSION || p.board_height > MAX_BOARD_DIMENSION) {
		printf("***ERROR*** Board size (%d x %d) exceeds maximum allowable dimensions\n", p.board_width, p.board_height); exit(-1);
	}
	p.num_pieces = GET_PARAM("NUM_PIECES", p.board_width);
	p.max_turns = GET_PARAM("MAX_TURNS", 10);
	p.board_size = p.board_width * p.board_height;	// number of cells on the board
	p.state_size = 2 * p.board_size;
	
	p.num_hidden = GET_PARAM("NUM_HIDDEN", 32);
	p.init_wgt_min = GET_PARAM("INIT_WGT_MIN", -.1);
	p.init_wgt_max = GET_PARAM("INIT_WGT_MAX", .1);
	
	p.alpha = GET_PARAMF("ALPHA", .20f);
	p.epsilon = GET_PARAMF("EPSILON", .10f);
	p.gamma = GET_PARAMF("GAMMA", .95f);
	p.lambda = GET_PARAMF("LAMBDA", .50f);
	
	p.num_agents = GET_PARAM("NUM_AGENTS", 64);
	p.num_sessions = GET_PARAM("NUM_SESSIONS", 16);	
	p.episode_length = GET_PARAM("EPISODE_LENGTH", 256);
	p.warmup_length = GET_PARAM("WARMUP_LENGTH", 256);
	
	p.run_on_CPU = PARAM_PRESENT("CPU");
	p.run_on_GPU = PARAM_PRESENT("GPU");
	if (!p.run_on_CPU && !p.run_on_GPU) {
		printf("***ERROR*** must specify at least one of --CPU and --GPU\n");
		exit(-1);
	}
	
	p.num_wgts = p.num_hidden * (2 * p.board_size + 3);	// number of weights for one agent
	p.num_agent_floats = (3*p.num_wgts + 3);	// total number of float values for an agent
												// (wgts, e, saved_wgts, alpha, epsilon, lambda)
	p.timesteps = p.num_sessions * p.num_agents * p.episode_length;
	p.agent_timesteps = p.timesteps * p.num_agents;
	
	p.champ = AGENT_FILE_CHAMP;
	
	printf("[FASTRACK][BOARD_SIZE %06d][NUM_PIECES %3d][MAX_TURNS %3d][SEED%10d][NUM_HIDDEN%4d][INIT_WGT_MIN%7.4f][INIT_WGT_MAX%7.4f][ALPHA%7.4f][EPSILON%7.4f][GAMMA%7.4f][LAMBDA%7.4f][NUM_AGENTS%7d][NUM_SESSIONS%7d][EPISODE_LENGTH%7d][WARMUP_LENGTH%7d]\n", 1000*p.board_width + p.board_height, p.num_pieces, p.max_turns, p.seed, p.num_hidden, p.init_wgt_min, p.init_wgt_max, p.alpha, p.epsilon, p.gamma, p.lambda, p.num_agents, p.num_sessions, p.episode_length, p.warmup_length);

	printf("\n");
	
	return p;
}

int main(int argc, const char **argv)
{
	PARAMS p = read_params(argc, argv);
	dump_params(p);
	
	// initialize agents on CPU and GPU
	AGENT *agCPU = init_agentsCPU(p);
//	dump_agentsCPU("after initialization", agCPU, 0);
	AGENT *agGPU = NULL;
	if (p.run_on_GPU) {
		agGPU = init_agentsGPU(agCPU);
	}

	// load champ weights for benchmark testing
	float *champ_wgts = load_champ(AGENT_FILE_CHAMP);
//	dump_agentsCPU("after load_champ", agCPU, 0);
	
	RESULTS *resultsCPU = NULL;
	RESULTS *resultsGPU = NULL;
	if (p.run_on_CPU) resultsCPU = runCPU(agCPU, champ_wgts);
	if (p.run_on_GPU) resultsGPU = runGPU(agGPU, champ_wgts);
	
	printf("done runs\n");

	if (resultsCPU) dumpResults(resultsCPU);
	if (resultsGPU) dumpResults(resultsGPU);

	printf("done dump results\n");
	
	// Save best agent to file from GPU run, or if no GPU run, from the CPU run
	if(p.run_on_GPU && resultsGPU) save_agent(AGENT_FILE_OUT, agGPU, resultsGPU->iBest);
	else if (p.run_on_CPU && resultsCPU) save_agent(AGENT_FILE_OUT, agCPU, resultsCPU->iBest);
	
	printf("saving agent\n");

	return 0;
}
