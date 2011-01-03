/*
 *  cuda_utils.cu
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include "cuda_utils.h"

#pragma mark timer functions

void CREATE_TIMER(unsigned int *p_timer){
	cutilCheckError(cutCreateTimer(p_timer)); 
	cutilCheckError(cutResetTimer(*p_timer));
}
void START_TIMER(unsigned int timer){ 
	cutilCheckError(cutResetTimer(timer));
	cutilCheckError(cutStartTimer(timer)); 
}
float STOP_TIMER(unsigned int timer, char *message){
	cutilCheckError(cutStopTimer(timer));
	float elapsed = cutGetTimerValue(timer);
	if (message) printf("%12.3f ms for %s\n", elapsed, message);
	return elapsed;
}
void PAUSE_TIMER(unsigned int timer){	cutilCheckError(cutStopTimer(timer));	}
void RESUME_TIMER(unsigned int timer){ cutilCheckError(cutStartTimer(timer));	}
void RESET_TIMER(unsigned timer){		cutilCheckError(cutResetTimer(timer));	}
void DELETE_TIMER(unsigned int timer){	cutilCheckError(cutDeleteTimer(timer)); }

void PRINT_TIME(float time, char *message)
{
	if (message) printf("%12.3f ms for %s\n", time, message);
}

#pragma mark device memory functions
float *device_copyf(float *data, unsigned count_data)
{
	float *d_data = NULL;
	unsigned size_data = count_data * sizeof(float);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_copyf] float host data at %p count = %d, ", data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size_data));
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("copied to %p\n", d_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
	return d_data;
}

unsigned *device_copyui(unsigned *data, unsigned count_data)
{
	unsigned *d_data = NULL;
	unsigned size_data = count_data * sizeof(unsigned);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_copyui] unsigned data at %p count = %d, ", data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size_data));
#ifdef TRACE_DEVICE_ALLOCATIONS
	printf("copied to 0x%p]\n", d_data);
#endif
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
	return d_data;
}

int *device_copyi(int *data, unsigned count_data)
{
	int *d_data = NULL;
	unsigned size_data = count_data * sizeof(int);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_copyi] integers data at %p count = %d, ", data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size_data));
#ifdef TRACE_DEVICE_ALLOCATIONS
	printf("copied to 0x%p]\n", d_data);
#endif
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
	return d_data;
}

float *device_allocf(unsigned count_data)
{
	float *d_data;
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_allocf] count = %d", count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, count_data * sizeof(float)));
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf(" at %p\n", d_data);
	#endif
	return d_data;
}

__global__ void fillui_kernel(unsigned *d_data, unsigned count, unsigned val)
{
	unsigned iGlobal = threadIdx.x + blockIdx.x * blockDim.x;
	if (iGlobal < count) d_data[iGlobal] = val;
}

unsigned *device_allocui(unsigned count_data)
{
	unsigned *d_data;
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_allocui] count = %d", count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, count_data * sizeof(unsigned)));
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf(" at %p\n", d_data);
	#endif
	return d_data;
}

unsigned *device_alloc_filledui(unsigned count_data, unsigned val)
{
	unsigned *d_data;
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_allocui] count = %d", count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, count_data * sizeof(unsigned)));
	dim3 blockDim(512);
	dim3 gridDim(1 + (count_data-1)/512);
	fillui_kernel<<<gridDim, blockDim>>>(d_data, count_data, val);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf(" at %p\n", d_data);
	#endif
	return d_data;
}

__global__ void device_fillf_kernel(float val, float *d_data, unsigned count)
{
	unsigned iGlobal = threadIdx.x + blockIdx.x * blockDim.x;
	if (iGlobal > count) return;
	d_data[iGlobal] = val;
}
void device_fillf(float val, float *d_data, unsigned count)
{
	dim3 blockDim(512);
	dim3 gridDim(1 + (count-1)/512);
	device_fillf_kernel<<<gridDim, blockDim>>>(val, d_data, count);
}


float *host_copyf(float *d_data, unsigned count_data)
{
	unsigned size_data = count_data * sizeof(float);
	float *data = (float *)malloc(size_data);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[host_copyf] float data at %p count = %d\n", d_data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(data, d_data, size_data, cudaMemcpyDeviceToHost));
	return data;
}

unsigned *host_copyui(unsigned *d_data, unsigned count_data)
{
	unsigned size_data = count_data * sizeof(unsigned);
	unsigned *data = (unsigned *)malloc(size_data);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[host_copyui] unsigned data at %p count = %d\n", d_data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(data, d_data, size_data, cudaMemcpyDeviceToHost));
	return data;
}

int *host_copyi(int *d_data, unsigned count_data)
{
	unsigned size_data = count_data * sizeof(int);
	int *data = (int *)malloc(size_data);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[host_copyi] int data at %p count = %d\n", d_data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(data, d_data, size_data, cudaMemcpyDeviceToHost));
	return data;
}

void host_dumpf(const char *str, float *data, unsigned nRows, unsigned nCols)
{
	printf("%s\n", str);
	printf("      ");
	for (int j = 0; j < nCols; j++) {
		printf("%7d   ", j);
	}
	printf("\n");
	for (int i = 0; i < nRows; i++) {
		printf("[%4d]", i);
		for (int j = 0; j < nCols; j++) {
			printf("%10.4f", data[i * nCols + j]);
		}
		printf("\n");
	}
}

void device_dumpf(const char *str, float *d_data, unsigned nRows, unsigned nCols)
{
	float *data = host_copyf(d_data, nRows * nCols);
	host_dumpf(str, data, nRows, nCols);
	if(data) free(data);
}

void host_dumpui(const char *str, unsigned *data, unsigned nRows, unsigned nCols)
{
	printf("%s\n", str);
	printf("      ");
	for (int j = 0; j < nCols; j++) {
		printf("%7d   ", j);
	}
	printf("\n");
	for (int i = 0; i < nRows; i++) {
		printf("[%4d]", i);
		for (int j = 0; j < nCols; j++) {
			printf("%10d", data[i * nCols + j]);
		}
		printf("\n");
	}
}

void host_dumpi(const char *str, int *data, unsigned nRows, unsigned nCols)
{
	printf("%s\n", str);
	printf("      ");
	for (int j = 0; j < nCols; j++) {
		printf("%7d   ", j);
	}
	printf("\n");
	for (int i = 0; i < nRows; i++) {
		printf("[%4d]", i);
		for (int j = 0; j < nCols; j++) {
			printf("%10d", data[i * nCols + j]);
		}
		printf("\n");
	}
}

void device_dumpui(const char *str, unsigned *d_data, unsigned nRows, unsigned nCols)
{
	unsigned *data = host_copyui(d_data, nRows * nCols);
	host_dumpui(str, data, nRows, nCols);
	if(data) free(data);
}

void device_dumpi(const char *str, int *d_data, unsigned nRows, unsigned nCols)
{
	int *data = host_copyi(d_data, nRows * nCols);
	host_dumpi(str, data, nRows, nCols);
	if(data) free(data);
}
