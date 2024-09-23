#include <chrono>
#include <iostream>

#include <xmmintrin.h>
#include <immintrin.h>

struct AlignedMemoryDeleter {
	void operator()(int32_t* ptr) {
		_aligned_free(ptr);
	};
};


std::unique_ptr<int32_t[], AlignedMemoryDeleter> allocateArray(int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(int32_t), alignment);

	// static_assert(memory % alignment == 0);

	const auto res_array = static_cast<int32_t*>(memory);
	return std::unique_ptr<int32_t[], AlignedMemoryDeleter>(res_array);
}

std::unique_ptr<int32_t[], AlignedMemoryDeleter> prepareArray(int startPoint, int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(int32_t), alignment);
	const auto res_array = static_cast<int32_t*>(memory);

	for (int i = 0; i < size; ++i) {
		res_array[i] = startPoint + i;
	}

	return std::unique_ptr<int32_t[], AlignedMemoryDeleter>(res_array);
}

void sum(
	const int32_t* const a,
	const int32_t* const b,
	int32_t* const res,
	const int size
	)
{
	for (int i = 0; i < size; i++) {
		res[i] = a[i] + b[i];
	}
}

void sumAvx(
	const int32_t* a,
	const int32_t* b,
	int32_t* res,
	const int size
	)
{
	const auto packLength = sizeof(__m512i) / sizeof(int);

	const auto max_i = (size / packLength) * packLength;

	for (int i = 0; i < max_i;i += packLength) {
		__m512i a_section = _mm512_load_si512 (reinterpret_cast<const __m512i*>(a + i));
		__m512i b_section = _mm512_load_si512 (reinterpret_cast<const __m512i*>(b + i));

		__m512i sum = _mm512_add_epi32(a_section, b_section);
		_mm512_store_si512(reinterpret_cast<__m256i*>(res + i), sum);
	}

	for (int i = max_i; i < size; ++i) {
		res[i] = a[i] + b[i];
	}
}

int main(int argc, char *argv[])
{
	const auto dataLength = 1024 * 1024 * 128;
	const auto a = prepareArray(1100, dataLength, 64);
	const auto b = prepareArray(1, dataLength, 64);

	const auto regular_res = allocateArray(dataLength, 64);
	const auto avx_res = allocateArray(dataLength, 64);

	auto start1 = std::chrono::high_resolution_clock::now();
	sum(a.get(), b.get(), regular_res.get(), dataLength);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> regular_duration = end1 - start1;

	auto start2 = std::chrono::high_resolution_clock::now();
	sumAvx(a.get(), b.get(), avx_res.get(), dataLength);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> simd_duration = end2 - start2;

	std::cout << "Regular " << regular_duration.count() << std::endl;
	std::cout << "SSE " << simd_duration.count() << std::endl;

	return 0;
}
