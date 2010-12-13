//
//  fastrack.h
//  fastrack
//
//  Created by Dwight Bell on 12/13/10.
//  Copyright dbelll 2010. All rights reserved.
//

#define MAX_BOARD_DIMENSION 8	// maximum value for width or length of board


// PARAMS structure is a convenient place to hold all dynamic parameters
typedef struct {
	// game parameters
	unsigned board_width;		// board width (power of 2)
	unsigned board_height;		// board height (power of 2)
	unsigned seed;				// seed value used to generate global seeds (which are then
								// used to generate each agents seeds)
	// agent parameters
	unsigned num_hidden;		// number of hidden nodes in nn (power of 2)
	float init_wgt_min;
	float init_wgt_max;
	
	// learning parameters
	float alpha;				// default learning rate
	float epsilon;				// default exploration rate
	float gamma;				// default discount rate
	float lambda;				// default lambda
	
	unsigned num_agents;		// number of agents in population (power of 2)
	unsigned episode_length;	// number of time steps in the learning period (power of 2)
	unsigned num_episodes;		// number of episodes in this run
	
	unsigned run_on_CPU;		// flags
	unsigned run_on_GPU;
	
	// calculated values
	unsigned board_size;		// board_width * board_height
	unsigned lg_bw;				// lg(board_width)
	unsigned lg_bh;				// lg(board_height)
	unsigned lg_ag;				// lg(num_agents);
	unsigned state_bits;		// 2*board_size = number of bits needed to encode a state
								//  = one bit for each square for each player
	unsigned state_ints;		// 1 + (state_bits-1) / 32 
								//  = number of unsigned ints needed to encode a state
	unsigned num_inputs;		// number of input nodes in nn = state_bits
	unsigned alloc_wgts;		// num_hidden * (2*board_size + 3) = space used for weight array
								// this value is the stride between agent's weight blocks
	unsigned agent_float_count;	// (2*alloc_wgts + 3) = total size of agent float data
								// wgt and W and all_wgts values and alpha, epsilon, and lambda
								// have 1 each
	
} PARAMS;


// AGENT structure is used to consolidate the pointers to all agent data.  Pointers may be
// all host pointers or all device pointers.
typedef struct{
	unsigned *seeds;		// random number seeds for each agent (num_agents * 4)
	float *wgts;		// nn wgts for each agent (num_agents * alloc_wgts)
	float *W;			// sum of lambda * gradient for each weight (num_agents * alloc_wgts)
	float *alpha;		// agent-specific alpha value (num_agents)
	float *epsilon;		// agent-specific epsilon value (num_agents)
	float *lambda;		// agent-specific lambda value (num_agents)
	
} AGENT;

typedef struct {
	unsigned *seeds;	// pointer to four seeds
	float *fdata;		// pointer to all float values
	unsigned num_wgts;	// number of weights (and of W)
	unsigned iWgts;		// index in fData to beginning of wgts
	unsigned iW;		// index in fData to beginning of @
	unsigned iAlpha;	// index in fData to alpha
	unsigned iEpsilon;	// index in fData to epsilon
	unsigned iLambda;	// index in fData to lambda
} COMPACT_AGENT;


// RESULTS holds pointers to the results of the learning, which are always on the CPU
typedef struct {
	unsigned allocated;	// number of allocated agents 
	COMPACT_AGENT *best;				// best agent after each episode
} RESULTS;



AGENT *init_agentsCPU(PARAMS p);
AGENT *init_agentsGPU(AGENT *agCPU);
void dump_agentsCPU(const char *str, AGENT *agCPU, unsigned dumpW);

void freeAgentCPU(AGENT *ag);
void freeCompactAgent(COMPACT_AGENT *ag);
void freeAgentGPU(AGENT *ag);

RESULTS *newResults(PARAMS *p);
void dumpResults(RESULTS *ag);
void freeResults(RESULTS *r);

unsigned *start_state();
void dump_state(unsigned *state);

RESULTS *runCPU(AGENT *ag);
RESULTS *runGPU(AGENT *ag);

