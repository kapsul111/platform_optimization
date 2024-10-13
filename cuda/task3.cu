#include <iostream>
#include <cuda.h>

__inline__ __device__ int warpReduceSum(int val) {
    // Full mask for active threads in a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

const int blockSize = 32;

int sum(int *a, int n) 
 {
	int result = 0;
	for (int i = 0; i < n; i++) {
		result += a[i];
	}

	return result;
 }

__global__ void sumBlockKernel(int *out, int *a, int n) 
{
	extern __shared__ int sharedData[];

	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;

	if (index < n) {
		sharedData[tid] = a[index];
	} else {
		sharedData[tid] = 0;
	}

	__syncthreads();
	for (int i = blockDim.x / 2;  i > 0; i /= 2) {
		if (tid < i) {
			sharedData[tid] += sharedData[tid + i];
		}

		__syncthreads();
	}

	if (tid == 0) {
		out[blockIdx.x] = sharedData[0];
	}
}

void sumGpu(int *out, int *a, int n, int blockNum, int threadNum)
{
	sumBlockKernel<<<blockNum, threadNum>>>(out, a, n); //to don`t copy whole a array and keep it

	blockNum = (blockNum + threadNum - 1) / threadNum;
	n = (n + threadNum - 1) / threadNum;

	while (n > 1) { 
		sumBlockKernel<<<blockNum, threadNum>>>(out, out, n);
		
		blockNum = (blockNum + threadNum - 1) / threadNum;
		n = (n + threadNum - 1) / threadNum;
	}
}


int main()
{
	constexpr auto N = 10000000;

	auto a = new int[N];
	
	for(int i = 0; i < N; i++) {
		a[i] = 2;
	}

	int *d_a = nullptr;
	int *d_out = nullptr;

	cudaMalloc((void**)&d_a, sizeof(int) * N);
	cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);

	const auto blockNumbers = (N + blockSize - 1) / blockSize;

	cudaMalloc((void**)&d_out, sizeof(int) * blockNumbers);

	sumGpu(d_out, d_a, N, blockNumbers, blockSize);

	int deviceResult = 0;
	cudaMemcpy(&deviceResult, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	const auto hostResult = sum(a, N);

	std::cout << "Host Result " << hostResult << std::endl;
	std::cout << "Device Result " << deviceResult << std::endl;

	if (hostResult == deviceResult) {
		std::cout << "Result is correct" << std::endl;
	} else {
		std::cout << "Result is WRONG!" << std::endl;
	}

	delete[] a;

	cudaFree(d_a);
	cudaFree(d_out);

	return 0;
}