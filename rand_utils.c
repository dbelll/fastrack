/*
 *  rand_utils.c
 *  fixnn
 *
 *  Created by Dwight Bell on 11/6/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include "rand_utils.h"

/* generate random number [0,1) */
float ranf() {
	return (float)random()/(float)RAND_MAX;
};

/* generate a random number between -max_mag and +max_mag */
float rand_wgt(float max_mag){
	return (2.0 * max_mag)*ranf() - max_mag;
};

float rand_wgt2(float minVal, float maxVal)
{
	return minVal + (maxVal - minVal)*ranf();
}


