#include <chrono>
#include <iostream>

#include <xmmintrin.h>
#include <immintrin.h>

struct AlignedMemoryDeleter {
	void operator()(char* ptr) {
		_aligned_free(ptr);
	};
};

using AlocatedMemory = std::unique_ptr<char[], AlignedMemoryDeleter>;


AlocatedMemory prepareString(const char *str, int size, int alignment)
{
	const auto memory = _aligned_malloc(size * sizeof(float), alignment);
	const auto res_str = static_cast<char*>(memory);

	memcpy(res_str, str, size);

	return AlocatedMemory(res_str);
}

int substringSearch(
	const char* const mainString,
	const int mainStringSize,
	const char* const substring,
	const int substringSize
	)
{
	if (!mainStringSize) {
		return -1;
	}

	if (!substringSize) {
		return 0;
	}

	for (int i = 0; i < (mainStringSize - substringSize) + 1; i++) {
		bool found = true;
		int k = i;
		for (int j = 0; j < substringSize; j++, k++) {
			if (mainString[k] != substring[j]) {
				found = false;
				break;
			}
		}
		if (found) {
			return i;
		}
	}
	return -1;
}

int substringAvxSearch(
	const char* const mainString,
	const int mainStringSize,
	const char* const substring,
	const int substringSize
	)
{
	if (!mainStringSize) {
		return -1;
	}

	if (!substringSize) {
		return 0;
	}

	const auto avx_register_size = 64;
	int max_avx_i = ((mainStringSize - substringSize + 1) / avx_register_size) * avx_register_size;

	__m512i firstLetter = _mm512_set1_epi8(substring[0]);

	int i = 0;
	for (; i < max_avx_i; i += avx_register_size) {
		__m512i section = _mm512_load_si512(reinterpret_cast<const __m512i*>(mainString + i));
		__mmask64 mask = _mm512_cmpeq_epi8_mask(section, firstLetter);

		if (mask == 0) {
			continue;
		}

		do {
			const auto matchPos = __lzcnt64(mask);

			const auto res = memcmp(
				reinterpret_cast<const void*>(mainString + i + matchPos),
				reinterpret_cast<const void*>(substring),
				substringSize
				);
			if (res == 0) {
				return i;
			}
			mask &= ~(1ULL << (avx_register_size - matchPos - 1));
		}
		while(mask > 0);
	}

	for (; i < (mainStringSize - substringSize) + 1; i++) {
		bool found = true;
		int k = i;
		for (int j = 0; j < substringSize; j++, k++) {
			if (mainString[k] != substring[j]) {
				found = false;
				break;
			}
		}
		if (found) {
			return i;
		}
	}

	return -1;
}

AlocatedMemory prepareMainString(int mainStrSize, int substrSize, int alignment)
{
	std::string mainString;

	int substringFitTimes = mainStrSize / substrSize;

	std::string regularStr;
	for (int i = 0; i < substrSize / 4; i++) {
		regularStr.push_back('a');
		regularStr.push_back('b');
		regularStr.push_back('c');
		regularStr.push_back('d');
	}

	for (int i = 0; i < substringFitTimes - 1; i++) {
		mainString.append(regularStr);
	}

	for (int i = 0; i < substrSize; i++) {
		mainString.push_back('c');
	}

	return prepareString(mainString.c_str(), mainStrSize, alignment);
}

AlocatedMemory prepareSubstring(int substrSize, int alignment)
{
	std::string substr;

	for (int i = 0; i < substrSize; i++) {
		substr.push_back('c');
	}

	return prepareString(substr.c_str(), substrSize, alignment);
}

int main(int argc, char *argv[])
{
	const auto mainStrLength = 1024 * 1024 * 128;
	const auto substrLength = 1024;
	const auto mainStr = prepareMainString(mainStrLength, substrLength, 64);
	const auto substr = prepareSubstring(substrLength, 64);

	auto start_search_loop = std::chrono::high_resolution_clock::now();

	const auto loopResult = substringSearch(mainStr.get(), mainStrLength, substr.get(), substrLength);

	auto end_search_loop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> loop_search_duration = end_search_loop - start_search_loop;

	auto avx_start_dot = std::chrono::high_resolution_clock::now();

	const auto avxResult = substringAvxSearch(mainStr.get(), mainStrLength, substr.get(), substrLength);

	auto avx_end_dot = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> avx_search_duration = avx_end_dot - avx_start_dot;

	std::cout << "Loop substring search time " << loop_search_duration.count() << std::endl;
	std::cout << "AVX substring search time " << avx_search_duration.count() << std::endl;

	std::cout << "Loop substring search result " << loopResult << std::endl;
	std::cout << "AVX substring search time result " << avxResult << std::endl;

	return 0;
}
