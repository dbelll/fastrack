//
//  fastrack.h
//  fastrack
//
//  Created by Dwight Bell on 12/13/10.
//  Copyright dbelll 2010. All rights reserved.
//

//#define USE_TEXTURE_FOR_MOVES

#define GLOBAL_WGTS_FORMAT 2
#define FILE_FORMAT 1

#define MAX_BOARD_DIMENSION 8	// maximum value for width or length of board, should be power of 2
#define MAX_BOARD_SIZE MAX_BOARD_DIMENSION * MAX_BOARD_DIMENSION
#define MAX_STATE_SIZE MAX_BOARD_SIZE * 2

#define LEARNING_LOG_FILE "standings.csv"
#define LEARNING_LOG_FILE_GPU "standingsGPU.csv"

#define AGFILE_GPU0 "GPU0.agent"		// filename for saved agent data
#define AGFILE_GPU1 "GPU1.agent"
#define AGFILE_GPU2 "GPU2.agent"
#define AGFILE_GPU3 "GPU3.agent"

#define AGFILE_CPU0 "CPU0.agent"
#define AGFILE_CPU1 "CPU1.agent"
#define AGFILE_CPU2 "CPU2.agent"
#define AGFILE_CPU3 "CPU3.agent"

#define AGENT_FILE_CHAMP00 "knight57n4v11.agent"
#define AGENT_FILE_CHAMP01 "GPU1n4_64v32_5x20000_pieces2_turns5_lambda025.agent"

#define REWARD_WIN 1.00f
#define REWARD_LOSS 0.00f
#define REWARD_TIME_LIMIT 0.50f

#define ALPHA_HALF_LIFE 20

enum OPPONENT_METHODS {
	OM_SELF=0,		// 0
	OM_FIXED1,		// 1
	OM_FIXED2,		// 2
	OM_BEST,		// 3
	OM_ONE,			// 4
};

// piece move definitions
#define MOVES_KNIGHT {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}
#define MOVES_ONE_MULTI {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}
#define MOVES_TWO_DIAG {{1, 1}, {2, 2}, {1, -1}, {2, -2}, {-1, -1}, {-2, -2}, {-1, 1}, {-2, 2}}
#define MOVES_JUMP_DIAG {{0, 1}, {2, 2}, {1, 0}, {2, -2}, {0, -1}, {-2, -2}, {-1, 0}, {-2, 2}}
#define MOVES_JUMP_ORTHO {{0, 2}, {1, 1}, {2, 0}, {1, -1}, {0, -2}, {-1, -1}, {-2, 0}, {-1, 1}}

#define MAX_MOVES 8			// the maximum number of possible moves in the piece move definitions

#define MAX_OPPONENTS 32

// macro for executing statement if variable show is true
// use before printf or other output statements to only print when show is TRUE:
// eg:		SHOW dump_state(state, turn, 1);
#define SHOW if(show)

// these macros are for accessing the boards within a compact state
#define X_BOARD(state) (state)
#define O_BOARD(state) ((state) + g_p.board_size)

#define X_BOARDGPU(state) (state)
#define O_BOARDGPU(state) ((state) + dc_board_size)

// macros to access one agent's values
#define AG_SEEDS(ag, iAg) (ag->seeds + iAg * 4 * g_p.board_size)
#define AG_WGTS(ag, iAg) (ag->wgts + iAg * g_p.wgts_stride)
#define AG_E(ag, iAg) (ag->e + iAg * g_p.wgts_stride)
#define AG_SAVED_WGTS(ag, iAg) (ag->saved_wgts + iAg * g_p.wgts_stride)
#define AG_ALPHA(ag, iAg) (ag->alpha + iAg)
#define AG_epsilon(ag, iAg) (ag->epsilon + iAg)
#define AG_LAMBDA(ag, iAg) (ag->lambda + iAg)
#define AG_STATE(ag, iAg) (ag->states + iAg * g_p.state_size)
#define AG_NEXT(ag, iAg) (ag->next_to_play + iAg * g_p.wgts_stride)

#define NAME_BUFF_SIZE 16	// buffer size for agent names

// PARAMS structure is a convenient place to hold all dynamic parameters
typedef struct {
	// game parameters
	unsigned board_width;		// board width (power of 2)
	unsigned board_height;		// board height (power of 2)
	unsigned num_pieces;		// pieces per side, random location at start of game,
								// value of 0 means used fixed starting position of first row
								// filled for each player
	unsigned max_turns;			// maximum turns per game
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
	
	unsigned num_agents;		// number of agents in population 
	unsigned num_sessions;		// number of learning sessions with num_agents episodes in each
	unsigned segs_per_session;	// segments per session
	unsigned episode_length;	// number of time steps in each learning episode
	unsigned warmup_length;		// number of time steps for initial training against RAND
	unsigned iChamp;			// index into the champs to be used for benchmarking
	unsigned benchmark_games;	// number of games to play vs. champ after each learning session
	unsigned benchmark_ops;		// number of opponents used for benchmarking
	unsigned benchmark_freq;	// the frequency of running the benchmark, in number of sessions
	unsigned standings_freq;	// the fequency of determining the standings and reseting agent w-l stats
	unsigned refresh_op_wgts_freq;	// the frequency for refreshing the opponents wgts used for learning
	unsigned determine_best_op_freq;		// the frequency to determine the best opponent
											// should be a multiple of the refresh_op_wgts_freq
	unsigned begin_using_best_ops;	// the session number to begin using the best opponents in learning sessions
	unsigned op_fraction;		// denominator of the fraction of agents that are used as opponents during learning
								// 2 ==> use 1/2 of agents, 4 ==> use 1/4, etc.
	unsigned num_opponents;		// the number of opponents each session
								// = num_agents / op_fraction
	enum OPPONENT_METHODS op_method;	// opponent assignment method
	unsigned half_opponents;	// largest power of 2 less than num_opponents
	unsigned half_benchmark_ops;
	unsigned *opgrid;			// the oppoents each agent will learn against (segs_per_session * num_opponents)
	unsigned *d_opgrid;			// device pointer to where opponent grid is stored in global memory
	unsigned run_on_CPU;		// flags
	unsigned run_on_GPU;
	
	// calculated values
	unsigned board_size;		// board_width * board_height
	unsigned state_size;		// 2 * board_size
	unsigned half_board_size;	// biggest power of two less than board_size, used for reductions
								// on the GPU
	unsigned board_bits;		// number of bits needed to cover board_size, ie lg(board_size) rounded up
	unsigned half_hidden;
	float piece_ratioX;			// num_pieces / board_size
	float piece_ratioO;			// num_pieces / (board_size - num_pieces)

	unsigned num_wgts;			// num_hidden * (2*board_size + 3) = space used for weight array
								// in compact format, used for shared memory
	unsigned wgts_stride;		// MAX_STATE_SIZE * (num_hidden + 2), this value is the stride
								// between weight blocks in the global memory layout
	unsigned num_agent_floats;	// (3*wgts_stride + 3) = total size of agent float global data
								// (wgts and e and  alpha, epsilon, and lambda)
	unsigned timesteps;			// num_sessions * num_agents * episode_length
	unsigned agent_timesteps;	// num_agents * timesteps
	
	const char *champ;			// name of file with champ's weights
} PARAMS;


// structure to hold the won-loss record of a competition of learning episode for an agent
typedef struct {
	unsigned agent;
	unsigned games;
	unsigned wins;
	unsigned losses;
} WON_LOSS;

// AGENT structure is used to consolidate the pointers to all agent data.  Pointers may be
// all host pointers or all device pointers.
typedef struct{
	unsigned *seeds;	// random number seeds for each agent (num_agents * 4 * board_size)
	float *wgts;		// nn wgts for each agent (num_agents * wgts_stride)
	float *e;			// eligibility trace (num_agents * wgts_stride)
	float *saved_wgts;	// saved copy of weights (num_agents * wgts_stride)
	float *delta_wgts;	// change in weights over last learning episode (num_agents * num_opponents * num_wgts)
	float *alpha;		// agent-specific alpha value (num_agents)
	float *epsilon;		// agent-specific epsilon value (num_agents)
	float *lambda;		// agent-specific lambda value (num_agents)
	unsigned *states;		// saved state information (num_agents * state_size)
	unsigned *next_to_play; // 0 ==> X, 1 ==> O (num_agents)
	WON_LOSS *wl;		// temporary area to store won-loss information prior to reducing
						// (num_agents * num_opponents * sizeof(WON_LOSS)
} AGENT;

//typedef struct {
//	unsigned *seeds;	// pointer to four seeds
//	float *fdata;		// pointer to all float values
//	unsigned num_wgts;	// number of weights (and of W)
//	unsigned iWgts;		// index in fData to beginning of wgts
//	unsigned iE;		// index in fData to beginning of eligibility trace
//	unsigned iAlpha;	// index in fData to alpha
//	unsigned iEpsilon;	// index in fData to epsilon
//	unsigned iLambda;	// index in fData to lambda
//} COMPACT_AGENT;


// RESULTS holds pointers to the results of the learning, which are always on the CPU
typedef struct {
	PARAMS p;				// global parameters used for this run
	WON_LOSS *standings;	// WON_LOSS structures for each agent after each learning session.
	WON_LOSS *vsChamp;		// WON_LOSS structures for each agent after each learning session with
							// the results of competing against the benchmark opponent
	unsigned iBest;			// index number for best agent this run
} RESULTS;

unsigned calc_num_wgts(unsigned num_hidden, unsigned board_size);
unsigned calc_wgts_stride(unsigned num_hidden, unsigned board_size);

AGENT *init_agentsCPU(PARAMS p);
AGENT *init_agentsGPU(AGENT *agCPU);
void dump_agentsCPU(const char *str, AGENT *agCPU, unsigned dumpW, unsigned dumpS);
float *load_champ(const char *file);
void save_agentsCPU(AGENT *agCPU, RESULTS *rCPU);
void save_agentsGPU(AGENT *agGPU, RESULTS *rGPU);

void freeAgentCPU(AGENT *ag);
void freeAgentGPU(AGENT *ag);

RESULTS *newResults(PARAMS *p);
void dumpResults(RESULTS *ag);
void dumpResultsGPU(RESULTS *ag);
void freeResults(RESULTS *r);
void freeResultsGPU(RESULTS *r);

unsigned *start_state();
void dump_state(unsigned *state);
//void dump_board(unsigned *board);

RESULTS *runCPU(AGENT *ag, float *champ_wgts);
RESULTS *runGPU(AGENT *ag, float *champ_wgts);

