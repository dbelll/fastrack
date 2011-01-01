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

