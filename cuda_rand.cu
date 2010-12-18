/*
	from http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
*/


/*
 *	Random nubmers on the GPU
 *
 *	
 *	float RandUniform(unsigned *seeds, unsigned stride);	// float, [0.0 1.0)
 *	unsigned RandUniformui(unsigned *seeds, unsigned stride);	// unsigned,  [0, RAND_MAX]
 *	float RandNormal(unsigned *seeds, unsigned stride);		// float, gaussian mean = 0 std = 1
 *
 *	seeds must point to 4 unsigned values with the specified stride
 */


__device__ __host__ inline unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
	unsigned b = (((z << S1) ^ z) >> S2);
	return z = (((z & M) << S3) ^ b);
}

__device__ __host__ inline unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
	return z = (A*z + C);
}

/* generate a random number, uses an array of 4 unsigned ints */
__device__ __host__ inline float HybridTaus(unsigned *z, unsigned stride)
{
	return 2.3283064365387e-10 * (float)(
		TausStep(z[0], 13, 19, 12, 4294967294UL) ^
		TausStep(z[stride], 2, 25, 4, 4294967288UL) ^
		TausStep(z[2*stride], 3, 11, 17, 4294967280UL) ^
		LCGStep(z[3*stride], 16654525, 1013904223UL)
	);
}

__device__ __host__ inline unsigned HybridTausui(unsigned *z, unsigned stride)
{
	return (
		TausStep(z[0], 13, 19, 12, 4294967294UL) ^
		TausStep(z[stride], 2, 25, 4, 4294967288UL) ^
		TausStep(z[2*stride], 3, 11, 17, 4294967280UL) ^
		LCGStep(z[3*stride], 16654525, 1013904223UL)
	);
}

/*
	Take two random [0,1) floats and produce 2 independent, gaussians with mean=0, var=1
*/
#define PI 3.14159265358979f
//__device__ inline void BoxMuller(float& u1, float& u2){
//    float   r = sqrtf(-2.0f * logf(u1));
//    float phi = 2 * PI * u2;
//    u1 = r * __cosf(phi);
//    u2 = r * __sinf(phi);
//}

__device__ __host__ inline void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * cosf(phi);
    u2 = r * sinf(phi);
}

__device__ __host__ inline float RandUniform(unsigned *z, unsigned stride)
{
	return HybridTaus(z, stride);
}

__device__ __host__ inline unsigned RandUniformui(unsigned *z, unsigned stride)
{
	return HybridTausui(z, stride);
}

__device__ __host__ inline float RandNorm(unsigned *z, unsigned stride)
{
	float u1 = RandUniform(z, stride);
	float u2 = RandUniform(z, stride);
    float r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    return r * cosf(phi);
//	return 0.123f;
}

