#include <iostream>
#include <random>

float random()
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1.0, 5.0);
    return distrib(gen);
}

 void addVectors(float *out, float *a, float *b, int n) 
 {
	for (int i = 0; i < n; i++) {
		out[i] = a[i] + b[i];
	}
 }

__global__ void addVectorsKernel(float *out, float *a, float *b, int n) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	out[index] = a[index] + b[index];
}

int main()
{
	constexpr auto N = 10000000;

	auto a = new float[N];
	auto b = new float[N];
	auto out_host = new float[N];
	auto out = new float[N];
	
	for(int i = 0; i < N; i++) {
		a[i] = random();
		b[i] = random();
	}

	float *d_a = nullptr;
	float *d_b = nullptr;
	float *d_out = nullptr;

	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_out, sizeof(float) * N);

	const auto blockSize = 256;
	const auto blockNumbers = (N + blockSize - 1) / blockSize;
	addVectorsKernel<<<blockNumbers, blockSize>>>(d_out, d_a, d_b, N);

	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	addVectors(out_host, a, b, N);

	const auto cmpresult = memcmp(out, out_host, N * sizeof(float));
	if (cmpresult == 0) {
		std::cout << "Result is correct" << std::endl;
	} else {
		std::cout << "Result is WRONG!" << std::endl;
	}

	delete[] a;
	delete[] b;
	delete[] out_host;
	delete[] out;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	return 0;
}