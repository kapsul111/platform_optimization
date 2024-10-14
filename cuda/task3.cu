#include <iostream>

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

	int value = sharedData[tid];
	for (int i = blockDim.x / 2;  i > 0; i /= 2) {
		value += __shfl_down_sync(0xFFFFFFFF, value, i);
	}

	if (tid == 0) {
		out[blockIdx.x] = value;
	}
}

void sumGpu(int *a, int n, int blockNum, int threadNum)
{
	while (n > 1) { 
		sumBlockKernel<<<blockNum, threadNum>>>(a, a, n);
		
		blockNum = (blockNum + threadNum - 1) / threadNum;
		n = (n + threadNum - 1) / threadNum;
	}
}

int main()
{
	constexpr auto N = 1000000000;

	auto a = new int[N];
	
	for(int i = 100; i < N; i++) {
		a[i] = i % 11;
	}

	int *d_a = nullptr;

	cudaMalloc((void**)&d_a, sizeof(int) * N);
	cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);

	const auto blockNumbers = (N + blockSize - 1) / blockSize;

	sumGpu(d_a, N, blockNumbers, blockSize);

	int deviceResult = 0;
	cudaMemcpy(&deviceResult, d_a, sizeof(int), cudaMemcpyDeviceToHost);

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

	return 0;
}