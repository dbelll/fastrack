/*
 *  helpers.c
 *  fastrack
 *
 *  Created by Dwight Bell on 12/13/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include "helpers.h"

/*
	log base 2 of an unsigned integer, rounded down
 */
unsigned log2ui(unsigned x)
{
	unsigned result = 0;
	while (x >= 2) {
		x >>= 1;
		++result;
	}
	return result;
}

// calculate the largest power of two less than x
unsigned halfpow2(unsigned x)
{
	unsigned result = 1;
	
	// keep doubling until we equal or exceed x
	while (result < x) result <<= 1;
	
	// back off the last doubling and return
	result >>= 1;
	return result;
}
