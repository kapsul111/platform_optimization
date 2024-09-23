#include <chrono>
#include <iostream>

#include <xmmintrin.h>
#include <immintrin.h>

struct AlignedMemoryDeleter {
	void operator()(float* ptr) {
		_aligned_free(ptr);
	}
};

void print(const float *const data, int size)
{
	for (int i = 0; i < size; i++) {
		std::cout << data[i] << " ";
	}
	std::cout << std::endl;
}

std::unique_ptr<float[], AlignedMemoryDeleter> allocateArray(int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(float), alignment);

	const auto res_array = static_cast<float*>(memory);
	return std::unique_ptr<float[], AlignedMemoryDeleter>(res_array);
}

std::unique_ptr<float[], AlignedMemoryDeleter> prepareArray(float startPoint, int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(float), alignment);
	const auto res_array = static_cast<float*>(memory);

	for (int i = 0; i < size; ++i) {
		res_array[i] = 1. + i;//startPoint + i + .5;
	}

	return std::unique_ptr<float[], AlignedMemoryDeleter>(res_array);
}

void transposedMatrix(
	const float* const in,
	float* const out,
	int size)
{
	const auto x_max = size;
	const auto y_max = size;
	for (int x = 0; x < x_max; x++) {
		for (int y = 0; y < y_max; y++) {
			out[x + y * size] = in[y + x * size];
		}
	}
}

void matrixProduct(
	const float* const a,
	const float* const b,
	const int size,
	float* const res
	)
{
	float tempRes = .0;

	for (int x = 0; x < size; x++) {
		for (int y = 0; y < size; y++) {
			tempRes = .0;
			for (int i = 0; i < size; i++) {
				tempRes += a[x * size + i] * b[i * size + y];
			}
			res[x * size + y] = tempRes;
		}
	}
}

void transposedMatrixProduct(
	const float* const a,
	const float* const b,
	const int size,
	float* const res
	)
{
	auto b_t = allocateArray(size * size, 64);

	transposedMatrix(b, b_t.get(), size);

	float tempRes = .0;
	for (int x = 0; x < size; x++) {
		for (int y = 0; y < size; y++) {
			tempRes = .0;
			for (int i = 0; i < size; i++) {
				tempRes += a[x * size + i] * b_t[y * size + i];
			}
			res[x * size + y] = tempRes;
		}
	}
}

void matrixProductAvx(
	const float* const a,
	const float* const b,
	const int size,
	float* const res
	)
{
	auto b_t = allocateArray(size * size, 64);
	transposedMatrix(b, b_t.get(), size);

	const auto packLength = sizeof(__m512) / sizeof(float);

	const auto max_avx_i = (size / packLength) * packLength;

	float tempRes = .0;
	for (int x = 0; x < size; x++) {
		for (int y = 0; y < size; y++) {
			__m512 sum = _mm512_setzero_ps();
			int i = 0;
			for (; i < max_avx_i; i += packLength) {
				__m512 a_section = _mm512_load_ps (reinterpret_cast<const __m512*>(a + i + x * size));
				__m512 b_section = _mm512_load_ps (reinterpret_cast<const __m512*>(b_t.get() + i + y * size));

				__m512 mul = _mm512_mul_ps(a_section, b_section);
				sum = _mm512_add_ps(sum, mul);
			}
			tempRes = _mm512_reduce_add_ps(sum);
			for (; i < size; ++i) {
				tempRes += a[x * size + i] * b_t[y * size + i];
			}
			res[x * size + y] = tempRes;
		}
	}
}

int main(int argc, char *argv[])
{
	std::cout << "Start" << std::endl;
	const long long matrixSize = 1024;
	const long long memorySize = matrixSize * matrixSize;
	const auto a = prepareArray(1, memorySize, 64);
	const auto b = prepareArray(1, memorySize, 64);

	const auto regular_res = allocateArray(memorySize, 64);
	const auto regular_transponse_res = allocateArray(memorySize, 64);
	const auto avx_res = allocateArray(memorySize, 64);

	auto start_loop = std::chrono::high_resolution_clock::now();
	matrixProduct(a.get(), b.get(), matrixSize, regular_res.get());
	auto end_loop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> loop_duration = end_loop - start_loop;

	auto start_tr_loop = std::chrono::high_resolution_clock::now();
	transposedMatrixProduct(a.get(), b.get(), matrixSize, regular_transponse_res.get());
	auto end_tr_loop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> loop_tr_duration = end_tr_loop - start_tr_loop;

	auto avx_start_dot = std::chrono::high_resolution_clock::now();
	matrixProductAvx(a.get(), b.get(), matrixSize, avx_res.get());
	auto avx_end_dot = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> avx_dot_duration = avx_end_dot - avx_start_dot;

	// std::cout << "Loop float matrix multiplication result" << std::endl;
	// print(regular_res.get(), memorySize);
	// std::cout << "Loop transponse float matrix multiplication result" << std::endl;
	// print(regular_transponse_res.get(), memorySize);
	// std::cout << "Avx float matrix multiplication result" << std::endl;
	// print(avx_res.get(), memorySize);

	std::cout << "Loop float matrix multiplication " << loop_duration.count() << std::endl;
	std::cout << "Loop float matrix  multiplication transponsed " << loop_tr_duration.count() << std::endl;
	std::cout << "Avx float matrix  multiplication transponsed " << avx_dot_duration.count() << std::endl;


	return 0;
}
