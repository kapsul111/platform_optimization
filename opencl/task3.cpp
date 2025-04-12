#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <vector>

void runLoop(double *result, const std::vector<double> &a)
{
	auto start = std::chrono::high_resolution_clock::now();

	*result = 0;
	for (int i = 0; i < a.size(); i++) {
		*result += a[i];
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	std::cout << "CPU loop solution time " << duration.count() << std::endl;
	std::cout << "CPU loop result: " << *result << std::endl;
}

const char *kernelSource = R"(

kernel void reduction(global double* out, global double *a, int size, __local double *temp)
{
	int globalId = get_global_id(0);
	int groupSize = get_local_size(0);
	int localId = get_local_id(0);

	if (globalId < size) {
		temp[localId] = a[globalId];
	} else {
		temp[localId] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = groupSize / 2; offset > 0; offset /= 2) {
		if (localId < offset) {
			temp[localId] += temp[localId + offset];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	int groupId = get_group_id(0);
	if (localId == 0) {
		out[groupId] = temp[0];
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

void runKernel(cl_device_type deviceType, const char *kernelName, double etalon, const std::vector<double> &a)
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

	cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size() * sizeof(double), NULL, &err);
	checkError(err, "clCreateBuffer d_a");

	cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, a.size() * sizeof(double), NULL, &err);
	checkError(err, "clCreateBuffer d_out");

	clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, a.size() * sizeof(double), a.data(), 0, NULL, NULL);

	const size_t blockSize = 16;

	err = clSetKernelArg(kernel, 0, sizeof(d_out), &d_out);
	checkError(err, "clSetKernelArg 0");

	err = clSetKernelArg(kernel, 1, sizeof(d_a), &d_a);
	checkError(err, "clSetKernelArg 1");

	int size = a.size();
	err = clSetKernelArg(kernel, 2, sizeof(size), &size);
	checkError(err, "clSetKernelArg 2");

	size_t localWorkSize = blockSize;
	size_t blocksNumber = (a.size() + blockSize - 1) / blockSize;
	size_t globalWorkSize = blocksNumber * blockSize;
	size_t localBufferSize = blockSize * sizeof(double);

	err = clSetKernelArg(kernel, 3, localBufferSize, NULL);
	checkError(err, "clSetKernelArg 3");

	cl_event event;

	clEnqueueNDRangeKernel(queue,
						   kernel,
						   1,
						   NULL,
						   &globalWorkSize,
						   &localWorkSize,
						   0,
						   NULL,
						   &event);

	while(globalWorkSize > 1) {
		size = blocksNumber;
		blocksNumber = (blocksNumber+ blockSize - 1) / blockSize;
		globalWorkSize = (globalWorkSize + blockSize - 1 ) / blockSize;

		err = clSetKernelArg(kernel, 0, sizeof(d_out), &d_out);
		checkError(err, "clSetKernelArg 0");

		err = clSetKernelArg(kernel, 1, sizeof(d_out), &d_out);
		checkError(err, "clSetKernelArg 1");

		err = clSetKernelArg(kernel, 2, sizeof(size), &size);
		checkError(err, "clSetKernelArg 2");

		err = clSetKernelArg(kernel, 3, localBufferSize, NULL);
		checkError(err, "clSetKernelArg 3");

		clEnqueueNDRangeKernel(queue,
							   kernel,
							   1,
							   NULL,
							   &globalWorkSize,
							   &localWorkSize,
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

	double result = 0;
	err = clEnqueueReadBuffer(queue,
						d_out,
						true,
						0,
						sizeof(result),
						&result,
						0,
						0,
						NULL);
	checkError(err, "clEnqueueReadBuffer");

	std::cout << std::endl;

	cl_ulong elapsedTime = endTime - startTime;

	std::cout << deviceTypeStr(deviceType) << " kernel profile info: time takes " << elapsedTime << std::endl;

	bool theSameAnswer = etalon == result;

	std::cout << deviceTypeStr(deviceType) << " kernel gives and CPU loop the same result: " << theSameAnswer << std::endl;
	std::cout << deviceTypeStr(deviceType) << " kernel result: " << result << std::endl;

	clReleaseMemObject(d_a);
	clReleaseMemObject(d_out);
}

int main() {
	size_t size = 1024 * 1024;

	double result = 0;

	std::vector<double> a(size);
	for (int i = 0; i < a.size(); i++) {
		a[i] = i % 5 + 1;
	}

	runLoop(&result, a);
	runKernel(CL_DEVICE_TYPE_CPU, "reduction", result, a);
	runKernel(CL_DEVICE_TYPE_GPU, "reduction", result, a);

	return 0;
}
