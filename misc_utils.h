/*
 *  misc_utils.h
 *  fixnn
 *
 *  Created by Dwight Bell on 11/6/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

/* return the index of the largest value */
int argmax(float *values, int n);

// timing feedback for long loops
// call timing_feedback_header with the total number of iterations
// call timing_feedback_dot with each iteration number
void timing_feedback_header(unsigned n);
void timing_feedback_dot(unsigned i);
