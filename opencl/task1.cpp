#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <vector>

void printFirstValues(const char *startMessage, int *out, int n)
{
	std::cout << startMessage << " ";
	for (int i = 0; i < n; i++) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}

void runLoop(std::vector<int> &out, const std::vector<int> &a, const std::vector<int> &b)
{
	auto start = std::chrono::high_resolution_clock::now();

	if (out.size() != a.size() || out.size() != b.size()) {
		std::cout << __FUNCTION__ << "wrong input size";
		return;
	}

	for (int i = 0; i < out.size(); i++) {
		out[i] = a[i] + b[i];
	}
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	std::cout << "CPU loop solution time " << duration.count() << std::endl;

	printFirstValues("CPU loop result first values:", out.data(), 10);
}

const char *kernelSource = R"(

kernel void add(global int* out, global int *a, global int *b, int N)
{
	int id = get_global_id(0);
	if (id < N) {
		out[id] = a[id] + b[id];
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

void runKernel(cl_device_type deviceType, const std::vector<int>& etalon, std::vector<int> &out, std::vector<int> &a, std::vector<int> &b, const size_t N)
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

	auto start = std::chrono::high_resolution_clock::now();
	cl_kernel kernel = clCreateKernel(program, "add", &err);
	checkError(err, "clCreateKernel");

	cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);
	checkError(err, "clCreateBuffer d_a");

	cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);
	checkError(err, "clCreateBuffer d_b");

	cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, &err);
	checkError(err, "clCreateBuffer d_b");

	clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, N * sizeof(int), a.data(), 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, N * sizeof(int), b.data(), 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(d_out), &d_out);
	checkError(err, "clSetKernelArg 0");

	err = clSetKernelArg(kernel, 1, sizeof(d_a), &d_a);
	checkError(err, "clSetKernelArg 1");

	err = clSetKernelArg(kernel, 2, sizeof(d_b), &d_b);
	checkError(err, "clSetKernelArg 2");

	int n_int = static_cast<int>(N);
	err = clSetKernelArg(kernel, 3, sizeof(n_int), &n_int);
	checkError(err, "clSetKernelArg 3");

	cl_event event;
	if (deviceType == CL_DEVICE_TYPE_GPU) {
		size_t localWorkSize = 256;
		size_t global_work_size = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;

		clEnqueueNDRangeKernel(queue,
							   kernel,
							   1,
							   NULL,
							   &global_work_size,
							   &localWorkSize,
							   0,
							   NULL,
							   &event);
	} else {
		clEnqueueNDRangeKernel(queue,
							   kernel,
							   1,
							   NULL,
							   &N,
							   NULL,
							   0,
							   NULL,
							   &event);
	}

	clFinish(queue);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << deviceTypeStr(deviceType) << " kernel solution time " << duration.count() << std::endl;

	cl_ulong startTime{};
	cl_ulong endTime{};

	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, nullptr);
	checkError(err, "clGetEventProfilingInfo start");

	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, nullptr);
	checkError(err, "clGetEventProfilingInfo end");

	err = clEnqueueReadBuffer(queue,
						d_out,
						true,
						0,
						out.size() * sizeof(int),
						out.data(),
						0,
						0,
						NULL);
	checkError(err, "clEnqueueReadBuffer");
	cl_int *resultValues = out.data();

	cl_ulong elapsedTime = endTime - startTime;

	std::cout << deviceTypeStr(deviceType) << " kernel profile info: time takes " << elapsedTime << std::endl;

	bool theSameAnswer = true;
	for (int i = 0; i < N; i++) {
		if (etalon[i] != resultValues[i]) {
			theSameAnswer = false;
		}
	}

	std::cout << deviceTypeStr(deviceType) << " kernel gives and CPU loop the same result: " << theSameAnswer << std::endl;
	const std::string printValueMessage = std::string(deviceTypeStr(deviceType)) +" kernel first values";
	printFirstValues(printValueMessage.c_str(), resultValues, 10);

	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_out);
}

int main() {
	const size_t N = 100000000;
	std::vector<int> a(N);
	std::vector<int> b(N);
	std::vector<int> out(N);
	std::vector<int> outGpu(N);
	std::vector<int> outCpu(N);

	for (int i = 0; i < N; i++) {
		a[i] = i % 5;
		b[i] = i % 5;
		out[i] = 0;
		outGpu[i] = 0;
		outCpu[i] = 0;
	}

	runLoop(out, a, b);
	runKernel(CL_DEVICE_TYPE_CPU, out, outCpu, a, b, N);
	runKernel(CL_DEVICE_TYPE_GPU, out, outCpu, a, b, N);

	return 0;
}
