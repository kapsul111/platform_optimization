#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

void printFirstValues(const char *startMessage, int *out, int n)
{
	std::cout << startMessage << " ";
	for (int i = 0; i < n; i++) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}

void runLoop(std::vector<int> &a)
{
	auto start = std::chrono::high_resolution_clock::now();

	std::sort(a.begin(), a.end());

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	std::cout << "CPU loop solution time " << duration.count() << std::endl;
	printFirstValues("CPU loop result first values:", a.data(), 10);
}

const char *kernelSource = R"(

kernel void sort(__global int* data, int j, int k) {
	int i = get_global_id(0);
	int ixj = i ^ j;

	if (ixj > i) {
		if ((i & k) == 0) {
			if (data[i] > data[ixj]) {
				int temp = data[i];
				data[i] = data[ixj];
				data[ixj] = temp;
			}
		} else {
			if (data[i] < data[ixj]) {
				int temp = data[i];
				data[i] = data[ixj];
				data[ixj] = temp;
			}
		}
	}
}

)";

void checkError(cl_int error, const char *message)
{
	if (error != CL_SUCCESS) {
		std::cerr << __LINE__ << " " << message << " Error code: " << error << std::endl;
	}
}

const char *deviceTypeStr(cl_device_type type)
{
	switch(type) {
	case CL_DEVICE_TYPE_CPU: return "CPU";
	case CL_DEVICE_TYPE_GPU: return "GPU";
	default: return "Unknown";
	}
}

void runKernel(cl_device_type deviceType, const char *kernelName, const std::vector<int> &etalon, std::vector<int> &a)
{
	cl_uint platformCount = 0;
	const int platformMax = 10;
	std::vector<cl_platform_id> platforms(platformMax);
	cl_int err = clGetPlatformIDs(platforms.size(), platforms.data(), &platformCount);
	checkError(err, "clGetPlatformIDs");

	platforms.resize(platformCount);

	cl_device_id device;
	for (auto platform : platforms) {
		err = clGetDeviceIDs(platform, deviceType, 1, &device, nullptr);
		if (err == CL_SUCCESS) {
			char platformName[128];
			err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
			checkError(err, "clGetPlatformInfo");

			std::string name(platformName);
			std::cout << "Found device " << deviceTypeStr(deviceType) << " on platform: " << name << std::endl;
			break;
		}
	}

	cl_context context = clCreateContext(NULL,
										 1,
										 &device,
										 NULL, NULL, &err);
	checkError(err, "clCreateContext");

	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl_command_queue queue = clCreateCommandQueue(context,
												  device,
												  properties,
												  &err);
	checkError(err, "clCreateCommandQueue");

	cl_program program = clCreateProgramWithSource(context,
												   1,
												   &kernelSource,
												   NULL,
												   &err);
	checkError(err, "clCreateProgramWithSource");

	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	checkError(err, "clBuildProgram");
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

	char *log = new char[log_size];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

	std::cout << "Build Log:\n" << log << std::endl;
	delete[] log;

	auto start = std::chrono::high_resolution_clock::now();
	cl_kernel kernel = clCreateKernel(program, kernelName, &err);
	checkError(err, "clCreateKernel");

	cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, a.size() * sizeof(int), NULL, &err);
	checkError(err, "clCreateBuffer d_a");

	clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, a.size() * sizeof(int), a.data(), 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(d_a), &d_a);
	checkError(err, "clSetKernelArg 0");

	size_t globalSize = a.size();
	for (int k = 2; k <= a.size(); k *= 2) {
		for (int j = k / 2; j > 0; j /= 2) {
			err = clSetKernelArg(kernel, 1, sizeof(int), &j);
			checkError(err, "clSetKernelArg 1");

			err = clSetKernelArg(kernel, 2, sizeof(int), &k);
			checkError(err, "clSetKernelArg 2");

			err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
			checkError(err, "clEnqueueNDRangeKernel");

			clFinish(queue);
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << deviceTypeStr(deviceType) << " kernel solution time " << duration.count() << std::endl;

	err = clEnqueueReadBuffer(queue,
						d_a,
						true,
						0,
						a.size() * sizeof(int),
						a.data(),
						0,
						0,
						NULL);
	checkError(err, "clEnqueueReadBuffer");

	bool theSameAnswer = etalon == a;

	std::cout << deviceTypeStr(deviceType) << " kernel gives and CPU loop the same result: " << theSameAnswer << std::endl;
	const std::string printValueMessage = std::string(deviceTypeStr(deviceType)) +" kernel first values";
	printFirstValues(printValueMessage.c_str(), a.data(), 10);

	clReleaseMemObject(d_a);
}

int main() {
	size_t size = 1024*1024;

	std::vector<int> a(size);
	for (int i = a.size() - 1; i >= 0; i--) {
		a[i] = a.size() - i;
	}

	std::vector<int> aCpu(size);
	for (int i = aCpu.size() - 1; i >= 0; i--) {
		aCpu[i] = aCpu.size() - i;
	}

	std::vector<int> aGpu(size);
	for (int i = aGpu.size() - 1; i >= 0; i--) {
		aGpu[i] = aGpu.size() - i;
	}

	runLoop(a);
	runKernel(CL_DEVICE_TYPE_CPU, "sort", a, aCpu);
	runKernel(CL_DEVICE_TYPE_GPU, "sort", a, aGpu);

	return 0;
}
