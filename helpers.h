/*
 *  helpers.h
 *  fastrack
 *
 *  Created by Dwight Bell on 12/13/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

unsigned log2ui(unsigned x);

// calculate the largest power of 2 less than x
unsigned halfpow2(unsigned x);

// number of bits needed for numbers in the range [0, x)
unsigned bits_needed(unsigned x);
