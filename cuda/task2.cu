#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>

const auto BLOCK_SIZE = 16;

#define CUDA_CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error)<< " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

float random()
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1.0, 5.0);
    return distrib(gen);
}

struct Matrix
{
	Matrix(int width, int height)
	: width(width)
	, height(height)
	, elements(new float[width * height])
	{
		const auto size = width * height;
		for (int i = 0; i < size; i++) {
			elements[i] = random();
		}	
	}

	Matrix(int width, int height, int initValue)
	: width(width)
	, height(height)
	, elements(new float[width * height])
	{
		std::fill(elements, elements + width * height, initValue);		
	}

	int width = 0;
	int height = 0;
	float *elements = nullptr;

	~Matrix()
	{
		delete[] elements;
	}
};

struct MatrixDevice
{
	int width = 0;
	int height = 0;
	int stride = 0;
	float *elements = nullptr;
};

__device__ float getElement(const MatrixDevice a, int row, int col)
{
	return a.elements[row * a.stride + col];
}

__device__ void setElement(const MatrixDevice a, int row, int col, float value)
{
	a.elements[row * a.stride + col] = value;
}

__device__ MatrixDevice getSubMatrix(const MatrixDevice a, int row, int col)
{
	MatrixDevice aSub {
		BLOCK_SIZE,
		BLOCK_SIZE,
		a.stride,
		&a.elements[row * BLOCK_SIZE * a.stride + col * BLOCK_SIZE]
		};

	return aSub;
}

void matrixMultiplication(Matrix &out, const Matrix &a, const Matrix &b) 
 {
	auto start = std::chrono::high_resolution_clock::now();
	if (a.width != b.height) {
		std::cout << "Wrong input matrix sizes" << std::endl;
		return;
	}

	if (out.width != b.width || out.height != a.height) {
		std::cout << "Wrong output matrix size" << std::endl;;
		return;
	}

	for (int row = 0; row < a.height; row++) {
		for (int col = 0; col < b.width; col++) {
			float temp = 0.;
			for (int i = 0; i < a.width; i++) {
				temp += a.elements[row * a.width + i] * b.elements[i * b.width + col];
			}
			out.elements[row * out.width + col] = temp;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
    std::cout << "CPU execution time: " << duration.count() << std::endl;
 }

__global__ void matrixMultiplicationKernel(MatrixDevice out, const MatrixDevice a, const MatrixDevice b) 
{
	const auto blockRow = blockIdx.y;
	const auto blockCol = blockIdx.x;

	MatrixDevice outBlock = getSubMatrix(out, blockRow, blockCol);

	const auto row = threadIdx.y;
	const auto col = threadIdx.x;

	float outValue = 0.;

	const auto blockCount = a.width / BLOCK_SIZE;
	for (int i = 0; i < blockCount; i++) {
		const MatrixDevice aSub = getSubMatrix(a, blockRow, i);
		const MatrixDevice bSub = getSubMatrix(b, i, blockCol);

		__shared__ float aTemp[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float bTemp[BLOCK_SIZE][BLOCK_SIZE];
		
		aTemp[row][col] = getElement(aSub, row, col);
		bTemp[row][col]  = getElement(bSub, row, col);

		__syncthreads();

		for (int el = 0; el < BLOCK_SIZE; el++) {
			outValue += aTemp[row][el] * bTemp[el][col];
		}
		__syncthreads();
	}

	setElement(outBlock, row, col, outValue);
}

int main()
{
	constexpr long a_width = 1024;
	constexpr auto a_height = 1024;
	constexpr auto b_width = 1024;
	constexpr auto b_height = a_width;

	constexpr auto a_size = a_width * a_height;
	constexpr auto b_size = b_width * b_height;
	constexpr auto out_size = b_width * a_height;

	Matrix a(a_width, a_height);
	Matrix b(b_width, b_height);
	Matrix out(b_width, a_height, 0.0f);	
	Matrix out_host_result(b_width, a_height, 0.0f);	

	const auto copyElementToCudaMemory = [](const Matrix &m, MatrixDevice &d_m, auto elementsNumbers) {
		const auto size = elementsNumbers * sizeof(float);
		CUDA_CHECK(cudaMalloc(&d_m.elements, size));
		CUDA_CHECK(cudaMemcpy(d_m.elements, m.elements, size, cudaMemcpyHostToDevice));
	};
	
	MatrixDevice d_a{ a.width, a.height, a.width, nullptr };
	copyElementToCudaMemory(a, d_a, a_size);
	
	MatrixDevice d_b{ b_width, b_height, b_width, nullptr };
	copyElementToCudaMemory(b, d_b, b_size);
	
	MatrixDevice d_out{ b_width, a_height, b_width, nullptr };	
	copyElementToCudaMemory(out, d_out, out_size);

	matrixMultiplication(out_host_result, a, b);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b.width / dimBlock.x, a.height / dimBlock.y);

	matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(d_out, d_a, d_b);
	CUDA_CHECK(cudaMemcpy(out.elements, d_out.elements, sizeof(float) * out_size, cudaMemcpyDeviceToHost));
	
	const auto cmpresult = memcmp(out.elements, out_host_result.elements, out_size * sizeof(float));
	if (cmpresult == 0) {
		std::cout << "Result is correct" << std::endl;
	} else {
		std::cout << "Result is WRONG!" << std::endl;
	}

	const auto printedPartSize = 2;
	for (int row = 0; row < printedPartSize; row++) {
		for (int col = 0; col < printedPartSize; col++) {
			std::cout << out_host_result.elements[row * out.width + col] << " ";
		}
		std::cout << std::endl;
	} 
	std::cout << std::endl;
	std::cout << "============================================" << std::endl;

	for (int row = 0; row < printedPartSize; row++) {
		for (int col = 0; col < printedPartSize; col++) {
			std::cout << out.elements[row * out.width + col] << " ";
		}
		std::cout << std::endl;
	} 
	std::cout << std::endl;

	CUDA_CHECK(cudaFree(d_a.elements));
	CUDA_CHECK(cudaFree(d_b.elements));
	CUDA_CHECK(cudaFree(d_out.elements));
	
	return 0;
}