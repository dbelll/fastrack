/*
 *  prototypes.h
 *  fastrack
 *
 *  Created by Dwight Bell on 1/1/11.
 *  Copyright 2011 dbelll. All rights reserved.
 *
 */

// prototypes
void freeAgentCPU(AGENT *ag);
void set_start_state(unsigned *state, unsigned pieces, unsigned *seeds, unsigned stride);
void dump_all_wgts(float *wgts, unsigned num_hidden);

int wl_compare(const void *p1, const void *p2);
int wl_byagent(const void *p1, const void *p2);
float winpct(WON_LOSS wl);
void set_agent_float_pointers(AGENT *ag);

void dump_agent(AGENT *agCPU, unsigned iag, unsigned dumpW, unsigned dumpSaved);
WON_LOSS *copy_standings_to_CPU(RESULTS *rGPU);
void print_standings(AGENT *ag, WON_LOSS *standings, WON_LOSS *vsChamp);
void dump_opponent_array();
