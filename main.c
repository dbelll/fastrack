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
	
	printf("  --NUM_HIDDEN          number of hidden nodes in agent nn");
	printf("  --INIT_THETA_MIN		minimum of range of possible initial weight values\n");
	printf("  --INIT_THETA_MAX		maximum of range of possible initial weight values\n");

	printf("  --ALPHA               float value for alpha, the learning rate parameter\n");
	printf("  --EPSILON             float value for epsilon, the exploration parameter\n");
	printf("  --GAMMA               float value for gamma, the discount factor\n");
	printf("  --LAMBDA              float value for lambda, the trace decay factor\n");
	
	printf("  --NUM_AGENTS			number of agents\n");
	printf("  --EPISODE_LENGTH      length of each learning episode\n");
	printf("  --NUM_EPISODES	    number of learning episodes\n");
	printf("  --CPU                 run on CPU");
	printf("  --GPU                 run on CPU");
	printf("  --HELP                print this help message\n");
	printf("default values will be used for any parameters not on command line\n");
}

void dump_params(PARAMS p)
{
	printf("game parameters:\n");
	printf("   seed         %10d\n", p.seed);
	printf("   board          (%3d x%3d)\n", p.board_width, p.board_height);
	printf("   state_bits      %7d\n", p.state_bits);
	printf("   state_ints      %7d\n", p.state_ints);
	printf("agent parameters:\n");
	printf("   num_inputs      %7d\n", p.num_inputs);
	printf("   num_hidden      %7d\n", p.num_hidden);
	printf("   alloc_wgts      %7d\n", p.alloc_wgts);
	printf("   init_wgt_min    %7.4f\n", p.init_wgt_min);
	printf("   init_wgt_max    %7.4f\n", p.init_wgt_max);
	printf("learning parameters:\n");
	printf("   num_agents      %7d\n", p.num_agents);
	printf("   episode_length  %7d\n", p.episode_length);
	printf("   num_episodes    %7d\n", p.num_episodes);
	printf("   alpha           %7.4f\n", p.alpha);
	printf("   epsilon         %7.4f\n", p.epsilon);
	printf("   gamma           %7.4f\n", p.gamma);
	printf("   lambda          %7.4f\n", p.lambda);
	printf("   run on CPU?     %s\n", p.run_on_CPU ? "TRUE" : "FALSE");
	printf("   run on GPU?     %s\n", p.run_on_GPU ? "TRUE" : "FALSE");
}

// read parameters from command line (or use default values) and print the header for this run
PARAMS read_params(int argc, const char **argv)
{
#ifdef VERBOSE
	printf("reading parameters...\n");
#endif
	
	PARAMS p;
	if (PARAM_PRESENT("HELP")) { display_help(); exit(0); }

	unsigned bs = GET_PARAM("BOARD_SIZE", 4004);
	p.board_width = bs / 1000;
	p.board_height = bs % 1000;
	if (p.board_width > MAX_BOARD_DIMENSION || p.board_height > MAX_BOARD_DIMENSION) {
		printf("***ERROR*** Board size (%d x %d) exceeds maximum allowable dimensions\n", p.board_width, p.board_height); exit(-1);
	}
	p.board_size = p.board_width * p.board_height;
	p.lg_bw = log2ui(p.board_width);
	p.lg_bh = log2ui(p.board_height);
	
	p.seed = GET_PARAM("SEED", 1000);
	p.num_hidden = GET_PARAM("NUM_HIDDEN", 32);
	
	p.init_wgt_min = GET_PARAM("INIT_WGT_MIN", -.1);
	p.init_wgt_max = GET_PARAM("INIT_WGT_MAX", .1);
	
	p.alpha = GET_PARAMF("ALPHA", .20f);
	p.epsilon = GET_PARAMF("EPSILON", .10f);
	p.gamma = GET_PARAMF("GAMMA", .95f);
	p.lambda = GET_PARAMF("LAMBDA", .50f);
	
	p.num_agents = GET_PARAM("NUM_AGENTS", 64);
	p.lg_ag = log2ui(p.num_agents);
	
	p.episode_length = GET_PARAM("EPISODE_LENGTH", 256);
	p.num_episodes = GET_PARAM("EPISODE_LENGTH", 16);
	
	p.run_on_CPU = PARAM_PRESENT("CPU");
	p.run_on_GPU = PARAM_PRESENT("GPU");
	if (!p.run_on_CPU && !p.run_on_GPU) {
		printf("***ERROR*** must specify at least one of --CPU and --GPU\n");
		exit(-1);
	}
	
	p.state_bits = 2 * p.board_size;
	p.state_ints = 1 + (p.state_bits - 1)/(8 * sizeof(unsigned));
	p.num_inputs = p.state_bits;
	p.alloc_wgts = p.num_hidden * (2 * p.board_size + 3);
	p.agent_float_count = (2*p.alloc_wgts + 3);
	
	printf("[FASTRACK][BOARD_SIZE %06d][SEED%10d][NUM_HIDDEN%4d][INIT_WGT_MIN%7.4f][INIT_WGT_MAX%7.4f][ALPHA%7.4f][EPSILON%7.4f][GAMMA%7.4f][LAMBDA%7.4f][NUM_AGENTS%7d][EPISODE_LENGTH%7d][NUM_EPISODES%7d]\n", 1000*p.board_width + p.board_height, p.seed, p.num_hidden, p.init_wgt_min, p.init_wgt_max, p.alpha, p.epsilon, p.gamma, p.lambda, p.num_agents, p.episode_length, p.num_episodes);

	// print flags
//	if (p.share_compete) printf("[SHARE_COMPETE]");
//	if (p.share_fitness) printf("[SHARE_FITNESS]");
	//	if (p.share_always) printf("[SHARE_ALWAYS]");
	printf("\n");
	
	
	return p;
}

int main(int argc, const char **argv)
{
	PARAMS p = read_params(argc, argv);
	dump_params(p);
	
	// initialize agents on CPU and GPU
	AGENT *agCPU = init_agentsCPU(p);
	AGENT *agGPU = NULL;
	if (p.run_on_GPU) {
		agGPU = init_agentsGPU(agCPU);
	}
	
//	dump_agentsCPU("initial agents", agCPU, 1);
	
	RESULTS *resultsCPU = NULL;
	RESULTS *resultsGPU = NULL;
	if (p.run_on_CPU) resultsCPU = runCPU(agCPU);
	if (p.run_on_GPU) resultsGPU = runGPU(agGPU);
	
	if (resultsCPU) dumpResults(resultsCPU);
	if (resultsGPU) dumpResults(resultsGPU);
	
	unsigned state[4];
	state[3] = 4278190080u;
	state[2] = 0;
	state[1] = 0;
	state[0] = 255;
	
	dump_state(start_state());
	
	return 0;
}
