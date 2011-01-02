//
//  fastrack.h
//  fastrack
//
//  Created by Dwight Bell on 12/13/10.
//  Copyright dbelll 2010. All rights reserved.
//


#define GLOBAL_WGTS_FORMAT 2
#define FILE_FORMAT 1

#define MAX_BOARD_DIMENSION 8	// maximum value for width or length of board, should be power of 2
#define MAX_BOARD_SIZE MAX_BOARD_DIMENSION * MAX_BOARD_DIMENSION
#define MAX_STATE_SIZE MAX_BOARD_SIZE * 2

#define LEARNING_LOG_FILE "standings.csv"
#define LEARNING_LOG_FILE_GPU "standingsGPU.csv"
#define AGENT_FILE_OUT "knight_out.agent"
#define AGENT_FILE_CHAMP "knight57n4v11.agent"
//#define CHAMP_GAMES 1000

#define REWARD_WIN 1.00f
#define REWARD_LOSS 0.00f
#define REWARD_TIME_LIMIT 0.50f

#define ALPHA_HALF_LIFE 20

// piece move definitions
#define MOVES_KNIGHT {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}
#define MOVES_ONE_MULTI {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}
#define MOVES_TWO_DIAG {{1, 1}, {2, 2}, {1, -1}, {2, -2}, {-1, -1}, {-2, -2}, {-1, 1}, {-2, 2}}
#define MOVES_JUMP_DIAG {{0, 1}, {2, 2}, {1, 0}, {2, -2}, {0, -1}, {-2, -2}, {-1, 0}, {-2, 2}}
#define MOVES_JUMP_ORTHO {{0, 2}, {1, 1}, {2, 0}, {1, -1}, {0, -2}, {-1, -1}, {-2, 0}, {-1, 1}}

#define MAX_MOVES 8			// the maximum number of possible moves in the piece move definitions

// macro for executing statement if variable show is true
// use before printf or other output statements to only print when show is TRUE:
// eg:		SHOW dump_state(state, turn, 1);
#define SHOW if(show)

// these macros are for accessing the boards within a compact state
#define X_BOARD(state) (state)
#define O_BOARD(state) ((state) + g_p.board_size)

#define X_BOARDGPU(state) (state)
#define O_BOARDGPU(state) ((state) + dc_board_size)

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
	unsigned episode_length;	// number of time steps in each learning episode
	unsigned warmup_length;		// number of time steps for initial training against RAND
	unsigned benchmark_games;	// number of games to play vs. champ after each learning session
	unsigned benchmark_freq;	// the frequency of running the benchmark, in number of sessions
	
	unsigned run_on_CPU;		// flags
	unsigned run_on_GPU;
	
	// calculated values
	unsigned board_size;		// board_width * board_height
	unsigned state_size;		// 2 * board_size
	unsigned half_board_size;	// biggest power of two less than board_size, used for reductions
								// on the GPU
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


// AGENT structure is used to consolidate the pointers to all agent data.  Pointers may be
// all host pointers or all device pointers.
typedef struct{
	unsigned *seeds;	// random number seeds for each agent (num_agents * 4 * board_size)
	float *wgts;		// nn wgts for each agent (num_agents * wgts_stride)
	float *e;			// eligibility trace (num_agents * wgts_stride)
	float *saved_wgts;	// saved copy of weights (num_agents * wgts_stride)
	float *alpha;		// agent-specific alpha value (num_agents)
	float *epsilon;		// agent-specific epsilon value (num_agents)
	float *lambda;		// agent-specific lambda value (num_agents)
	unsigned *states;		// saved state information (num_agents * state_size)
	unsigned *next_to_play; // 0 ==> X, 1 ==> O (num_agents)
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


// structure to hold the won-loss record of a competition of learning episode for an agent
typedef struct {
	unsigned agent;
	unsigned games;
	unsigned wins;
	unsigned losses;
} WON_LOSS;

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
void save_agent(const char *file, AGENT *agCPU, unsigned iAg);

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

