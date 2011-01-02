/*
 *  main.h
 *  fastrack
 *
 *  Created by Dwight Bell on 12/13/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

/*
	Flags for controlling the output
 
	#define the value to enable it's corresponding output
*/

#define NO_OUTPUT			// define this symbol to turn off all debugging output


#ifndef NO_OUTPUT

#define VERBOSE

//#define DUMP_MOVES
//#define DUMP_ALL_AGENT_UPDATES
//#define SHOW_SAMPLE_GAMES_AFTER_LEARNING 4

//#define DUMP_INITIAL_AGENTS
#define DUMP_FINAL_AGENTS_CPU
#define DUMP_FINAL_AGENTS_GPU
//#define DUMP_CHAMP_WGTS

#define DUMP_SAVED_WGTS 0				// 1 => dump saved weights, 0 => don't dump them

#define AGENT_DUMP_BOARD_ONLY

#endif
