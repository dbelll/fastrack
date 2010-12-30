/*
 *  cuda_utils.h
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

/*
 *	Utilities for...
 *		Reading command line parameters
 *			GET_PARM(<param>, <default_value>)	// get an integer parameter
 *			GET_PARMF(<param>, <default_value>)	// get a float parameter
 *			PARAM_PRESENT(<param>)				// check for existance of a parameter
 *
 *		
 *
 *		Allocating and transferring data to/from device
 *			device_copyf(<h_data>, <count>)		// copy an array to the device
 *			device_copyui(<h_data>, <count>)	
 *												
 *			device_allocf(<count>)				// allocate device memory for <count> values
 *			device_allocui(<count>)				
 *
 *			host_copyf(<d_data>, <count>)		// allocate host memory and copy from device
 *			host_copyui(<d_data>, <count>)
 *
 *		Dumping data arrays from host or device
 *			host_dumpf("message", <h_data>, <rows>, <cols>);
 *			host_dumpui("message", <h_data>, <rows>, <cols>);
 *			device_dumpf("message", <d_data>, <rows>, <cols>);		** NOT YET IMPLEMENTED **
 *			device_dumpui("message", <d_data>, <rows>, <cols>);		** NOT YET IMPLEMENTED **
 *
 *		Timers...
 *			CREAT_TIMER(&timer)
 *			START_TIMER(timer)
 *			STOP_TIMER(timer, "message")
 *			DELETE_TIMER(timer)
 */

#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#pragma mark -
#pragma mark Flags to turn stuff on and off
// define this symbol to print a message at all device_copyx calls
//#define TRACE_DEVICE_ALLOCATIONS

// define these symbols to turn on pre-kernel messages and post-kernel error checking
#define __PRE_KERNEL_ON
#define __POST_KERNEL_ON

static int __iTemp;			// used in GET_PARAM macros
static float __fTemp;		// used in GET_PARAM macros

#pragma mark -
#pragma mark memory allocating, dumping, copying

// macros to read command line arguments or use a default value
// This macro assums argc and argv are their normal values found in main()
#define GET_PARAM(str, default) (CUTTrue == cutGetCmdLineArgumenti(argc, argv, (str), &__iTemp)) ? __iTemp : (default)
#define GET_PARAMF(str, default) (CUTTrue == cutGetCmdLineArgumentf(argc, argv, (str), &__fTemp)) ? __fTemp : (default)
#define PARAM_PRESENT(str) (CUTTrue == cutCheckCmdLineFlag(argc, argv, (str)))

// allocate room on the device and copy data from host, returning the device pointer
// returned pointer must be ultimately freed on the device
float *device_copyf(float *data, unsigned count);
unsigned *device_copyui(unsigned *data, unsigned count);
int *device_copyi(int *data, unsigned count);

// allocate room on the device, returning the device pointer
// returned pointer must be ultimately freed on the device
float *device_allocf(unsigned count);
unsigned *device_allocui(unsigned count);
unsigned *device_alloc_filledui(unsigned count, unsigned val);

void device_fillf(float val, float *d_data, unsigned count);

// allocate room on the host and copy data from device, returning the host pointer
// returned pointer must be ultimately freed on the host
float *host_copyf(float *d_data, unsigned count);
unsigned *host_copyui(unsigned *d_data, unsigned count);
int *host_copyui(int *d_data, unsigned count);


void host_dumpf(const char *str, float *data, unsigned nRows, unsigned nCols);
void device_dumpf(const char *str, float *data, unsigned nRows, unsigned nCols);
void host_dumpui(const char *str, unsigned *data, unsigned nRows, unsigned nCols);
void host_dumpi(const char *str, int *data, unsigned nRows, unsigned nCols);
void device_dumpui(const char *str, unsigned *data, unsigned nRows, unsigned nCols);
void device_dumpi(const char *str, int *data, unsigned nRows, unsigned nCols);

#ifdef TRACE_DEVICE_ALLOCATIONS

#define deviceFree(d_data) printf("[device_free] data at %p\n", d_data);	\
							cudaFree(d_data);

#else

#define deviceFree(d_data) cudaFree(d_data)

#endif


// Macros for calculating timing values.
// Caller must supply a pointer to unsigned int when creating a timer,
// and the unsigned int for other timer calls.
void CREATE_TIMER(unsigned int *p_timer);
void START_TIMER(unsigned int timer);
float STOP_TIMER(unsigned int timer, char *message);
void DELETE_TIMER(unsigned int timer);
void PAUSE_TIMER(unsigned int timer);
void RESUME_TIMER(unsigned int timer);
void RESET_TIMER(unsigned timer);
void DELETE_TIMER(unsigned int timer);
void PRINT_TIME(float time, char *message);


/*
 *		Use Cuda Events to get precise GPU timings.
 *		These timings will be consistent with the profiler timing values.
 *
 *		First, declare a float variabile to hold the elapsed time
 *		Use CUDA_EVENT_PREPARE before doing any timing to setup variables and create the events
 *		Use CUDA_EVENT_START before launching the kernel.
 *		Use CUDA_EVENT_STOP(t) after launching the kernel, where t is the float used to accumulate time
 *		Time values can be printed in a consistent format by calling PRINT_TIME(t, "timing message");
 *		When all timing is done, use CUDA_EVENT_CLEANUP once all event timing is done.
 */

#define CUDA_EVENT_PREPARE	cudaEvent_t __start, __stop;	\
							float __timeTemp = 0.0f;		\
							cudaEventCreate(&__start);		\
							cudaEventCreate(&__stop);

#define CUDA_EVENT_START	cudaEventRecord(__start, 0);
#define CUDA_EVENT_STOP(t)	cudaEventRecord(__stop, 0);							\
							cudaEventSynchronize(__stop);							\
							cudaEventElapsedTime(&__timeTemp, __start, __stop);	\
							t += __timeTemp;

#define CUDA_EVENT_CLEANUP	cudaEventDestroy(__start);	\
							cudaEventDestroy(__stop);

#define CUDA_EVENT_STOP2(t, str)	CUDA_EVENT_STOP(t);		\
									CUT_CHECK_ERROR(#str" execution failed!");


// CUDA_EVENTN_... macros can be used to manage n timers
#define CUDA_EVENTN_PREPARE(n)	cudaEvent_t __start2[n], __stop2[n];	\
									float __timeTemp2[n];				\
									unsigned __numTemp2 = n;			\
									for(int i=0; i<(n); i++){			\
										cudaEventCreate(__start2 + i);	\
										cudaEventCreate(__stop2 + i);}	\

#define CUDA_EVENTN_START(i)	cudaEventRecord(__start2[i], 0);

#define CUDA_EVENTN_STOP(t, i)	cudaEventRecord(__stop2[i], 0);								\
								cudaEventSynchronize(__stop2[i]);							\
								cudaEventElapsedTime(__timeTemp2+i, __start2[i], __stop2[i]);	\
								t += __timeTemp2[i];

#define CUDA_EVENTN_CLEANUP		for(int i=0; i<__numTemp2; i++){	\
									cudaEventDestroy(__start2[i]);	\
									cudaEventDestroy(__stop2[i]);}

/*
 *	PRE_ and POST_KERNEL macros which can be turned on and off from #define's
 *	
 *	PRE_KERNEL macro assumes the block and grid dimensions are in variables blockDim and gridDim
 *	If using different variable names, use the PRE_KERNEL2 macro
 */
#ifdef __PRE_KERNEL_ON

#define PRE_KERNEL(str) printf("about to call %s with block size (%d x %d) and grid size (%d x %d)\n", str, blockDim.x, blockDim.y, gridDim.x, gridDim.y);

#define PRE_KERNEL2(str, bd, gd) printf("about to call %s with block size (%d x %d) and grid size (%d x %d)\n", str, bd.x, bd.y, gd.x, gd.y);

#else

#define PRE_KERNEL(str)
#define PRE_KERNEL2(str, bd, gd)

#endif


#ifdef __POST_KERNEL_ON

#define POST_KERNEL(str)	CUT_CHECK_ERROR(#str" execution failed!!\n");	\
							printf(#str" execution succeeded\n")

#else

#define POST_KERNEL(str)

#endif



#endif

