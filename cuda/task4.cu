#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int random()
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 5);
    return distrib(gen);
}

void sortHost(int *data, int N)
{
	auto start = std::chrono::high_resolution_clock::now();

	std::sort(data, data + N);
	
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
    std::cout << "CPU execution time: " << duration.count() << std::endl;
}

void sortDevice(int *data, int N)
{
	thrust::device_vector<int> deviceData(data, data + N);
	thrust::sort(deviceData.begin(), deviceData.end());
	thrust::copy(deviceData.begin(), deviceData.end(), data);
}

int main()
{
	constexpr auto N = 10000000;

	auto a = new int[N];
	for(int i = 0; i < N; i++) {
		a[i] = random();
	}

	auto aDevice = new int[N];
	for (int i = 0; i < N; i++) {
		aDevice[i] = a[i];
	}
	std::memcpy(aDevice, a, sizeof(int) * N);

	sortHost(a, N);
	sortDevice(aDevice, N);

	const auto cmpResult = std::memcmp(aDevice, a, sizeof(int) * N);
	if (cmpResult == 0) {
		std::cout << "Result is correct" << std::endl;
	} else {
		std::cout << "Result is WRONG!" << std::endl;
	}

	for (int i = 0; i < 10; i ++) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
	
	for (int i = 0; i < 10; i ++) {
		std::cout << aDevice[i] << " ";
	}
	std::cout << std::endl;

	delete[] a;
	delete[] aDevice;

	return 0;
}