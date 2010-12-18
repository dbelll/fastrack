/*
 *  misc_utils.c
 *  fixnn
 *
 *  Created by Dwight Bell on 11/6/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include <stdio.h>
#include "misc_utils.h"

/* return the index of the largest value */
int argmax(float *values, int n)
{
	int i, max = 0;
	
	for(i=1; i<n; i++){
		if(values[i] > values[max]) max = i;
	}
	return max;
}


// helper functions to print a timing indicator to stdout
static int _k_ = 1;
//static int _n_minus_one_ = 1;
void timing_feedback_header(unsigned n)
{
	_k_ = 1;
	if (n > 40) {
		_k_ = (1 + (n-1)/40);
	}
	for (int i = 0; i < (n/_k_); i++) {
		printf("-");
	}
//	_n_minus_one_ = (n / _k_) - 1;
	printf("|\n");
}

void timing_feedback_dot(unsigned i)
{
	if (0 == (i+1) % _k_) { printf("."); fflush(NULL); }
//	if (i == _n_minus_one_) printf("\n");
}
