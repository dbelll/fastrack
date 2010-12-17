/*
 *  play.c
 *  fastrack
 *
 *  Created by Dwight Bell on 12/16/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 *
 *	Compete against the computer.
 */

#include "play.h"



int main(int argc, const char *argv)
{
	if (argc != 2) {
		printf("usage: play <agent_file>\n");
		exit(-1);
	}
	
	FILE *f = fopen(argv[1], "r");
	if (!f) {
		printf("*** ERROR *** could not open agent file %s\n", argv[1]);
		exit(-1);
	}
	
	
}