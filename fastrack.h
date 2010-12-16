//
//  fastrack.h
//  fastrack
//
//  Created by Dwight Bell on 12/13/10.
//  Copyright dbelll 2010. All rights reserved.
//

#define MAX_BOARD_DIMENSION 8	// maximum value for width or length of board

// piece move definitions
#define MOVES_KNIGHT {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}
#define MOVES_ONE_MULTI {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}
#define MOVES_TWO_DIAG {{1, 1}, {2, 2}, {1, -1}, {2, -2}, {-1, -1}, {-2, -2}, {-1, 1}, {-2, 2}}
#define MOVES_JUMP_DIAG {{0, 1}, {2, 2}, {1, 0}, {2, -2}, {0, -1}, {-2, -2}, {-1, 0}, {-2, 2}}
#define MOVES_JUMP_ORTHO {{0, 2}, {1, 1}, {2, 0}, {1, -1}, {0, -2}, {-1, -1}, {-2, 0}, {-1, 1}}

#define MAX_STATE_SIZE 4
#define MAX_BOARD_INTS 2

#define MAX_MOVES 8			// the maximum number of possible moves in the piece move definitions

#define X_BOARD(state) (state)
#define O_BOARD(state) ((state) + g_p.board_size)

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
	unsigned state_size;		// 2 * board_size
//	unsigned lg_bw;				// lg(board_width)
//	unsigned lg_bh;				// lg(board_height)
//	unsigned lg_ag;				// lg(num_agents);
//	unsigned board_bits;		// = board_size, bits needed to encode one player's position
//	unsigned board_ints;		// 1 + (board_bits-1) / 32
								//    = unsigned ints needed to encode one player's position
//	unsigned board_unused;		// hi-order bits not used in the highest int in teh board 
								//		= 32 * board_ints - board_bits
//	unsigned state_size;		// 2 * board_ints
								//  = number of unsigned ints used to encode a state
//	unsigned num_inputs;		// number of input nodes in nn = state_bits
	unsigned num_wgts;			// num_hidden * (2*board_size + 3) = space used for weight array
								// this value is the stride between agent's weight blocks
	unsigned num_agent_floats;	// (2*alloc_wgts + 3) = total size of agent float data
								// (wgts and e and  alpha, epsilon, and lambda)
	
} PARAMS;


// AGENT structure is used to consolidate the pointers to all agent data.  Pointers may be
// all host pointers or all device pointers.
typedef struct{
//	unsigned *state;	// state (num_agents * g_p.state_size)
	unsigned *seeds;	// random number seeds for each agent (num_agents * 4)
	float *wgts;		// nn wgts for each agent (num_agents * num_wgts)
	float *e;			// eligibility trace (num_agents * num_wgts)
	float *alpha;		// agent-specific alpha value (num_agents)
	float *epsilon;		// agent-specific epsilon value (num_agents)
	float *lambda;		// agent-specific lambda value (num_agents)
	
} AGENT;

typedef struct {
	unsigned *seeds;	// pointer to four seeds
	float *fdata;		// pointer to all float values
	unsigned num_wgts;	// number of weights (and of W)
	unsigned iWgts;		// index in fData to beginning of wgts
	unsigned iE;		// index in fData to beginning of eligibility trace
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
void dump_board(unsigned *board);

RESULTS *runCPU(AGENT *ag);
RESULTS *runGPU(AGENT *ag);

