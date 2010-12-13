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

static unsigned g_seeds[4] = {0, 0, 0, 0};
static PARAMS g_p;

#pragma mark -
#pragma mark misc.

/*
	Initialize the global seeds using specified seed value.
*/
void set_global_seeds(unsigned seed)
{
	srand(seed);
	for (int i = 0; i < 4; i++) {
		g_seeds[i] = rand();
	}
}

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
		if (ag->W) CUDA_SAFE_CALL(cudaFree(ag->W));
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
		if (ag->W) free(ag->W);
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
float val_for_state(float *wgts, int *state, float *hidden)
{
	float out = 0.0f;
	
	for (unsigned iHidden = 0; iHidden < g_p.num_hidden; iHidden++) {
		// first add in the bias
		hidden[iHidden] = -1.0f * wgts[iHidden];
		
		// next, loop over all bits in the state, and add in their contribution
		unsigned iWgt = g_p.num_hidden;	// index into wgts for idx = 0
		for (int i = 0; i < g_p.state_ints; i++) {
			unsigned s = state[i];
			while (s) {
				if (s & 1) hidden[iHidden] += wgts[iHidden + iWgt];
				s >>= 1;
				iWgt += g_p.num_hidden;
			}
		}
		
		// next, apply the activation function
		hidden[iHidden] = sigmoid(hidden[iHidden]);
		
		// now add this hidden node's contribution to the output
		out += hidden[iHidden] * wgts[iHidden + iWgt];
	}
	
	// finally, add the bias to the output value and apply the sigmoid function
	out += -1.0f * wgts[g_p.alloc_wgts - g_p.num_hidden];
	return sigmoid(out);
}

// return a pointer to the starting state for the game
unsigned *start_state()
{
	static unsigned *ss = NULL;
	if (ss == NULL) {
		ss = (unsigned *)calloc(g_p.state_ints, sizeof(unsigned));
		unsigned first_row = (1 << g_p.board_width)-1;
		unsigned last_row = first_row << (32 - g_p.board_width);
		ss[0] = first_row;
		ss[g_p.state_ints-1] |= last_row;
	}
	return ss;
}

#pragma mark -
#pragma mark dump stuff

// Calculate the row value, from the given state for the specified player
unsigned rowbits(unsigned *state, unsigned player, unsigned row)
{
//	printf("rowbits for player %u, row %u\n", player, row);
	unsigned row_bit = player * g_p.board_size + row * g_p.board_width;
//	printf("row_bit is %u\n", row_bit);
	unsigned iState = row_bit / 32;
	unsigned iRowStart = row_bit % 32;
//	printf("iState is %u and iRowStart is %u\n", iState, iRowStart);
	unsigned rb = state[iState] & (((1 << g_p.board_width)-1) << iRowStart);
//	printf("bits before shifting: %u", rb);
	rb >>= row_bit;
//	printf("... bits after shifting: %u\n", rb);
	return rb;
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

void dump_state(unsigned *state)
{
//	printf("dump_state for %u %u %u %u\n", state[3], state[2], state[1], state[0]);
	dump_col_header(3, g_p.board_width);
	for (int r = g_p.board_height - 1; r >= 0; r--) {
		printf("%2u ", r+1);
//		printf("row %d\n", r);
		unsigned s0 = rowbits(state, 0, r);
		unsigned s1 = rowbits(state, 1, r);
//		printf("player 0 row bits is %d\n", s0);
//		printf("player 1 row bits is %d\n", s1);
		unsigned count = 0;
		while (s0 || s1) {
//			printf("s0 = %u  s1 = %u\n", s0, s1);
			if ((s0 & 1) && (s1 & 1)) {
				printf("***ERROR*** inconsistent board\n");
				exit(-1);
			}
			if (s0 & 1) {
				printf(" X");
			}else if (s1 & 1) {
				printf(" O");
			}else {
				printf(" .");
			}
			s0 >>= 1;
			s1 >>= 1;
			count += 1;
		}
		while (count++ < g_p.board_width) {
			printf(" .");
		}
		printf("%3u", r+1);
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
		float *pW = agCPU->W + iag * g_p.alloc_wgts;
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


void dumpResults(RESULTS *r)
{
	printf("Best agents each round...\n");
	for (int i = 0; i < r->allocated; i++) {
		if (r->best+i){
			printf("[ROUND%3d]\n", i);
			dump_compact_agent(r->best+i);
		}
	}
}




#pragma mark -
#pragma mark CPU - Only
RESULTS *newResults()
{
	RESULTS *r = (RESULTS *)malloc(sizeof(RESULTS));
	r->allocated = g_p.num_episodes;
	r->best = (COMPACT_AGENT *)malloc(r->allocated * sizeof(COMPACT_AGENT));
	return r;
}

void freeResults(RESULTS *r)
{
	if (r) {
		for (int i = 0; i < r->allocated; i++)
			if (r->best + i) freeCompactAgent(r->best+i);
	
		free(r);
	}
}

// calculates agent pointers based on offset from ag->wgts
void set_agent_float_pointers(AGENT *ag)
{
	ag->W = ag->wgts + g_p.alloc_wgts * g_p.num_agents;
	ag->alpha = ag->W + g_p.alloc_wgts * g_p.num_agents;
	ag->epsilon = ag->alpha + g_p.num_agents;
	ag->lambda = ag->epsilon + g_p.num_agents;
}

AGENT *init_agentsCPU(PARAMS p)
{
	printf("init_agentsCPU...\n");
	// save the parameters
	g_p = p;

	// allocate and initialize the agent data on CPU
	AGENT *ag = (AGENT *)malloc(sizeof(AGENT));
	unsigned count = 4*p.num_agents;
	ag->seeds = (unsigned *)malloc(count * sizeof(unsigned));
	for (int i = 0; i < 4*p.num_agents; i++) ag->seeds[i] = rand();
	
//	printf("seeds allocated at %p\n", ag->seeds);
	
	// allocate one chunk of float data and set up pointers to appropriate parts of that chunk
	count = (2*p.alloc_wgts + 3) * p.num_agents;
	ag->wgts = (float *)malloc(count * sizeof(float));
	set_agent_float_pointers(ag);
	
//	printf("total of %d float values allocated\n", count);
//	printf("float values allocated at %p, first value is %f\n", ag->wgts, ag->wgts[0]);
//	printf("other pointers ag->W is %p\n", ag->W);
//	printf("           ag->alpha is %p\n", ag->alpha);
//	printf("         ag->epsilon is %p\n", ag->epsilon);
//	printf("          ag->lambda is %p\n", ag->lambda);
	
	// initialize values
	printf("initializing weights for %d values with min of %f and max of %f\n", p.alloc_wgts * p.num_agents, p.init_wgt_min, p.init_wgt_max);
	for (int i=0; i < p.alloc_wgts * p.num_agents; i++){
//		printf("%d ", i);
		ag->wgts[i] = rand_wgt2(p.init_wgt_min, p.init_wgt_max);
//		printf("- ");
		ag->W[i] = 0.0f;
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

RESULTS *runCPU(AGENT *agCPU)
{
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




