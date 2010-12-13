/*
 *  rand_utils.h
 *  fixnn
 *
 *  Created by Dwight Bell on 11/6/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include <stdlib.h>
#include <math.h>

/* generate random number [0,1) */
float ranf();

/* generate a random number between -max_mag and +max_mag */
float rand_wgt(float max_mag);

float rand_wgt2(float minVal, float maxVal);



