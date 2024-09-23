#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

#include <xmmintrin.h>
#include <immintrin.h>

std::vector<int> generateData(int startPoint, int size)
{
	std::vector<int> res(size);

	int number = startPoint;

	std::generate(res.begin(), res.end(), [&number] {
		return number++;
	});

	return res;
}

void sum(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &res)
{
	for (int i = 0; i < a.size(); i++) {
		res[i] = a[i] + b[i];
	}
}

void sumSse(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &res)
{
	int i = 0;

	const auto packLength = sizeof(__m128i) / sizeof(int);

	for (; i < (res.size() - packLength) ;i += packLength) {
		__m128i a_section = _mm_load_si128 (reinterpret_cast<const __m128i*>(&a[i]));
		__m128i b_section = _mm_load_si128 (reinterpret_cast<const __m128i*>(&b[i]));

		__m128i sum = _mm_add_epi32(a_section, b_section);
		_mm_store_si128(reinterpret_cast<__m128i*>(&res[i]), sum);
	}

	for (; i < a.size(); ++i) {
		res[i] = a[i] + b[i];
	}
}

void sumAvx(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &res)
{
	int i = 0;

	const auto packLength = sizeof(__m512i) / sizeof(int);

	for (; i < (res.size() - packLength) ;i += packLength) {
		__m512i a_section = _mm512_loadu_si512 (reinterpret_cast<const __m512i*>(&a[i]));
		__m512i b_section = _mm512_loadu_si512 (reinterpret_cast<const __m512i*>(&b[i]));

		__m512i sum = _mm512_add_epi32(a_section, b_section);
		_mm512_store_si512(reinterpret_cast<__m256i*>(&res[i]), sum);
	}

	for (; i < a.size(); ++i) {
		res[i] = a[i] + b[i];
	}
}

int main(int argc, char *argv[])
{
	const auto dataLength = 1024*1024*128;
	const auto a = generateData(1100, dataLength);
	const auto b = generateData(1, dataLength);

	std::vector<int> regularSum(dataLength);

	{
		auto start = std::chrono::high_resolution_clock::now();

		sum(a, b, regularSum);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> regular_duration = end - start;

		std::cout << "Regular " << regular_duration.count() << std::endl;
	}

	{
		std::vector<int> sseSum(dataLength);

		auto start = std::chrono::high_resolution_clock::now();

		sumSse(a, b, sseSum);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> simd_duration = end - start;

		std::cout << "SSE " << simd_duration.count() << std::endl;
		std::cout << "The same results " << (regularSum == sseSum) << std::endl;
	}

	{
		std::vector<int> avxSum(dataLength);

		auto start = std::chrono::high_resolution_clock::now();

		sumAvx(a, b, avxSum);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> avxDuration = end - start;

		std::cout << "AVX " << avxDuration.count() << std::endl;
		std::cout << "The same results " << (regularSum == avxSum);
	}


	return 0;
}
