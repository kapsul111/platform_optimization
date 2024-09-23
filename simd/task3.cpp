#include <chrono>
#include <iostream>

#include <xmmintrin.h>
#include <immintrin.h>

struct AlignedMemoryDeleter {
	void operator()(float* ptr) {
		_aligned_free(ptr);
	};
};


std::unique_ptr<float[], AlignedMemoryDeleter> allocateArray(int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(float), alignment);

	const auto res_array = static_cast<float*>(memory);
	return std::unique_ptr<float[], AlignedMemoryDeleter>(res_array);
}

std::unique_ptr<float[], AlignedMemoryDeleter> prepareArray(int startPoint, int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(float), alignment);
	const auto res_array = static_cast<float*>(memory);

	for (int i = 0; i < size; ++i) {
		res_array[i] = startPoint + i + .5;
	}

	return std::unique_ptr<float[], AlignedMemoryDeleter>(res_array);
}

void sum(
	const float* const a,
	const float* const b,
	float* const res,
	const int size
	)
{
	for (int i = 0; i < size; i++) {
		res[i] = a[i] + b[i];
	}
}

double dotProduct(
	const float* const a,
	const float* const b,
	const int size
	)
{
	double res = .0;
	for (int i = 0; i < size; i++) {
		res += a[i] * b[i];
	}

	return res;
}

void sumAvx(
	const float* a,
	const float* b,
	float* res,
	const int size
	)
{
	const auto packLength = sizeof(__m512) / sizeof(float);

	const auto max_i = (size / packLength) * packLength;

	for (int i = 0; i < max_i;i += packLength) {
		__m512 a_section = _mm512_load_ps (reinterpret_cast<const __m512*>(a + i));
		__m512 b_section = _mm512_load_ps (reinterpret_cast<const __m512*>(b + i));

		__m512 sum = _mm512_add_ps(a_section, b_section);
		_mm512_store_ps(reinterpret_cast<__m256*>(res + i), sum);
	}

	for (int i = max_i; i < size; ++i) {
		res[i] = a[i] + b[i];
	}
}

double dotProductAvx(
	const float* a,
	const float* b,
	const int size
	)
{
	const auto packLength = sizeof(__m512) / sizeof(float);

	const auto max_avx_i = (size / packLength) * packLength;

	__m512 sum = _mm512_setzero_ps();

	for (int i = 0; i < max_avx_i; i += packLength) {
		__m512 a_section = _mm512_load_ps (reinterpret_cast<const __m512*>(a + i));
		__m512 b_section = _mm512_load_ps (reinterpret_cast<const __m512*>(b + i));

		__m512 mul = _mm512_mul_ps(a_section, b_section);
		sum = _mm512_add_ps(sum, mul);
	}

	double res = _mm512_reduce_add_ps(sum);

	for (int i = max_avx_i; i < size; ++i) {
		res += a[i] * b[i];
	}

	return res;
}

int main(int argc, char *argv[])
{
	const auto dataLength = 1024 * 1024 * 128;
	const auto a = prepareArray(1100, dataLength, 64);
	const auto b = prepareArray(1, dataLength, 64);

	const auto loop_sum_res = allocateArray(dataLength, 64);
	const auto avx_sum_res = allocateArray(dataLength, 64);

	auto loop_addition_start = std::chrono::high_resolution_clock::now();
	sum(a.get(), b.get(), loop_sum_res.get(), dataLength);
	auto loop_addition_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> loop_duration = loop_addition_end - loop_addition_start;

	auto start2 = std::chrono::high_resolution_clock::now();
	sumAvx(a.get(), b.get(), avx_sum_res.get(), dataLength);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> simd_duration = end2 - start2;

	const auto sameSumResult = 0 == memcmp(static_cast<void*>(loop_sum_res.get()), static_cast<void*>(avx_sum_res.get()), dataLength * sizeof(float));
	std::cout << "Loop float addition " << loop_duration.count() << std::endl;
	std::cout << "AVX float addition " << simd_duration.count() << std::endl;

	std::cout << "Loop and AVX give the same sum result " << sameSumResult << std::endl;
	std::cout << "===================================================================" << std::endl;

	auto start_dot_loop = std::chrono::high_resolution_clock::now();
	const auto loop_dot_multiplication_res = dotProduct(a.get(), b.get(), dataLength);
	auto end_dot_avx = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> loop_dot_duration = end_dot_avx - start_dot_loop;

	auto avx_start_dot = std::chrono::high_resolution_clock::now();
	const auto avx_dot_multiplication_res = dotProductAvx(a.get(), b.get(), dataLength);
	auto avx_end_dot = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> avx_dot_duration = avx_end_dot - avx_start_dot;

	std::cout << "Loop float dot multiplication " << loop_dot_duration.count() << std::endl;
	std::cout << "AVX float dot multiplication " << avx_dot_duration.count() << std::endl;

	std::cout << "Loop float dot multiplication result" << loop_dot_multiplication_res << std::endl;
	std::cout << "AVX float dot multiplication result" << avx_dot_multiplication_res << std::endl;

	return 0;
}
