/*
 *  wgt_pointers.h
 *  fastrack
 *
 *  Created by Dwight Bell on 1/1/11.
 *  Copyright 2011 dbelll. All rights reserved.
 *
 */

/*
 Accessors for the wgts (and e's) 
 There are separate accessors for global memory and shared memory.  In global memory,
 the weights are aligned to 128 byte boundaries for faster access.  In shared memory, the
 weights are stored more compactly.
 Weight names:
 BH	bias -> hidden
 IH	input -> hidden
 HO	hidden -> output
 BO	bias -> output
 */
//float *pBH(float *w, unsigned iH){ return w + iH; }			// bias to hidden
//float *pIH(float *w, unsigned iI, unsigned iH){ return w + iH + g_p.num_hidden * (iI + 1);}	// in -> hidden
//float *pHO(float *w, unsigned iH){ return w + g_p.num_hidden * (1 + g_p.state_size) + iH; } // hidden -> out
//float *pBO(float *w){ return w + g_p.num_wgts - g_p.num_hidden; }	// bias to output

#if GLOBAL_WGTS_FORMAT == 1
/*
 version 1 - compact storage for global and shared memory
 B->H	(num_hidden)
 I->H	(state_size * num_hidden)
 H->O	(num_hidden)
 B->O	1
 Weights padded out to be (state_size + 3) * num_hidden values for both shared and global.
 */
float *pgBH(float *w, unsigned iH){ return w + iH; }
float *pgIH(float *w, unsigned iI, unsigned iH){ return w + iH + g_p.num_hidden * (iI + 1);}
float *pgHO(float *w, unsigned iH){ return w + g_p.num_hidden * (1 + g_p.state_size) + iH; }
float *pgBO(float *w){ return w + g_p.wgts_stride - g_p.num_hidden; }

float *psBH(float *w, unsigned iH){ return w + iH; }
float *psIH(float *w, unsigned iI, unsigned iH){ return w + iH + g_p.num_hidden * (iI + 1);}
float *psHO(float *w, unsigned iH){ return w + g_p.num_hidden * (1 + g_p.state_size) + iH; }
float *psBO(float *w){ return w + g_p.wgts_stride - g_p.num_hidden; }

#elif GLOBAL_WGTS_FORMAT == 2
/*
 version 2 - global has I->H first with rows corresponding to each hidden node and colums
 padded out to 2 * MAX_STATE_SIZE.  The two boards start at offset 0 and MAX_BOARD_SIZE.
 The B->H and H->O values are next and each is padded out to 2 * MAX_STATE_SIZE columns
 The single value B->O is in the last value of the H->O area.
 I->H	(num_hidden * MAX_STATE_SIZE)
 B->H	(MAX_STATE_SIZE)
 H->O	(MAX_STATE_SIZE-1)
 B->O	1
 Global weights take up (MAX_STATE_SIZE + 2) * num_hidden + 1 values.
 
 Compact format removes all padding:
 I->H	(num_hidden * state_size)
 B->H	num_hidden
 H->O	num_hidden
 B->O	1
 Total size for compact format is (state_size + 2)*num_hidden + 1	
 */

float *pgIXH(float *w, unsigned iX, unsigned iH){ return w + iH * MAX_STATE_SIZE + iX; }
float *pgIOH(float *w, unsigned iO, unsigned iH){ return w + iH * MAX_STATE_SIZE + MAX_BOARD_SIZE + iO; }
float *pgIH(float *w, unsigned iI, unsigned iH){ 
	return (iI >= g_p.board_size) ? pgIOH(w, iI-g_p.board_size, iH) : pgIXH(w, iI, iH); 
}
float *pgBH(float *w, unsigned iH){ return w + g_p.num_hidden * MAX_STATE_SIZE + iH; }
float *pgHO(float *w, unsigned iH){ return w + (1 + g_p.num_hidden) * MAX_STATE_SIZE + iH; }
float *pgBO(float *w){ return w + g_p.wgts_stride - 1; }

float *psIH(float *w, unsigned iI, unsigned iH){ return w + iH * g_p.state_size + iI; }
float *psBH(float *w, unsigned iH){ return w + g_p.state_size * g_p.num_hidden + iH; }
float *psHO(float *w, unsigned iH){ return w + g_p.state_size * (g_p.num_hidden + 1) + iH; }
float *psBO(float *w){ return w + g_p.num_wgts - 1; }
// macros for use on device
// They evaluate to a pointer to the first value for the specified section of weights
#define G_IXH(w, iH) (w + iH * MAX_STATE_SIZE)
#define G_IOH(w, iH) (G_IXH(w, iH) + MAX_BOARD_SIZE)
#define G_BH(w) (w + dc_num_hidden * MAX_STATE_SIZE)
#define G_HO(w) (G_BH(w) + MAX_STATE_SIZE)
#define G_BO(w) (G_HO(w) + MAX_STATE_SIZE - 1)

#define S_IXH(w, iH) (w + iH * dc_state_size)
#define S_IOH(w, iH) (S_IXH(w, iH) + dc_board_size)
#define S_BH(w) (w + dc_num_hidden * dc_state_size)
#define S_HO(w) (S_BH(w) + dc_num_hidden)
#define S_BO(w) (S_HO(w) + dc_num_hidden)

#elif GLOBAL_WGTS_FORMAT == 3

/*
 version 3 - global has I->H first with rows corresponding to each hidden node and columns = state_size.
 The B->H and H->O values are next and each is num_hidden columns.
 The single value B->O is in the last value of the H->O area.
 Total size is padded to a multiple of 32.
 I->H	(num_hidden * state_size)
 B->H	(num_hidden)
 H->O	(num_hidden)
 B->O	1
 Global weights take up (state_size + 2) * num_hidden + 1 values, rounded up to multiple of 32.
 
 Compact format removes padding at the end:
 I->H	(num_hidden * state_size)
 B->H	num_hidden
 H->O	num_hidden
 B->O	1
 Total size for compact format is (state_size + 2)*num_hidden + 1	
 */

float *pgIXH(float *w, unsigned iX, unsigned iH){ return w + iH * g_p.state_size + iX; }
float *pgIOH(float *w, unsigned iO, unsigned iH){ return w + iH * g_p.state_size + g_p.board_size + iO; }
float *pgIH(float *w, unsigned iI, unsigned iH){ 
	return (iI >= g_p.board_size) ? pgIOH(w, iI-g_p.board_size, iH) : pgIXH(w, iI, iH); 
}

float *psIH(float *w, unsigned iI, unsigned iH){ return w + iH * g_p.state_size + iI; }
float *psBH(float *w, unsigned iH){ return w + g_p.state_size * g_p.num_hidden + iH; }
float *psHO(float *w, unsigned iH){ return w + g_p.state_size * (g_p.num_hidden + 1) + iH; }
float *psBO(float *w){ return w + g_p.num_wgts - 1; }

float *pgBH(float *w, unsigned iH){ return psBH(w, iH); }
float *pgHO(float *w, unsigned iH){ return psHO(w, iH); }
float *pgBO(float *w){ return psBO(w); }


// macros for use on device
// They evaluate to a pointer to the first value for the specified section of weights
#define G_IXH(w, iH) (w + iH * dc_state_size)
#define G_IOH(w, iH) (G_IXH(w, iH) + dc_board_size)
#define G_BH(w) (w + dc_num_hidden * dc_state_size)
#define G_HO(w) (G_BH(w) + dc_num_hidden)
#define G_BO(w) (G_HO(w) + dc_num_hidden)

#define S_IXH(w, iH) (w + iH * dc_state_size)
#define S_IOH(w, iH) (S_IXH(w, iH) + dc_board_size)
#define S_BH(w) (w + dc_num_hidden * dc_state_size)
#define S_HO(w) (S_BH(w) + dc_num_hidden)
#define S_BO(w) (S_HO(w) + dc_num_hidden)



#endif

float gBH(float *w, unsigned iH){ return *pgBH(w, iH);}
float gIH(float *w, unsigned iI, unsigned iH){ return *pgIH(w, iI, iH); }
float gHO(float *w, unsigned iH){ return *pgHO(w, iH); }
float gBO(float *w){ return *pgBO(w); }
float sBH(float *w, unsigned iH){ return *psBH(w, iH);}
float sIH(float *w, unsigned iI, unsigned iH){ return *psIH(w, iI, iH); }
float sHO(float *w, unsigned iH){ return *psHO(w, iH); }
float sBO(float *w){ return *psBO(w); }
