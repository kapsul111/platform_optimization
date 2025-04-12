#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <vector>

struct Matrix
{
	Matrix(int width, int height)
		: width(width)
		, height(height)
		, elements(new float[width * height])
	{
		const auto size = width * height;
		for (int i = 0; i < size; i++) {
			elements[i] = 1;
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

void printFirstValues(const char *startMessage, float *out, int n)
{
	std::cout << startMessage << " ";
	for (int i = 0; i < n; i++) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}

void runLoop(Matrix &out, const Matrix& a, const Matrix &b)
{
	if (out.width != b.width ||
		out.height != a.height ||
		a.width != b.height) {
		std::cerr << __FUNCTION__ << " wront input and output matrix sizes";
		return;
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (int column = 0; column < out.width; column++) {
		for (int row = 0; row < out.height; row++) {
			float value = 0.;
			for (int i = 0; i < a.width; i++) {
				value += a.elements[a.width * row + i] * b.elements[b.width * i + column];
			}
			out.elements[out.width * row + column] = value;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	std::cout << "CPU loop solution time " << duration.count() << std::endl;

	printFirstValues("CPU loop result first values:", out.elements, 10);
}

const char *kernelSource = R"(

kernel void multiplyMatrix(global float* out, global float *a, global float *b, int bWidth, int aWidth)
{
	int column = get_global_id(0);
	int row = get_global_id(1);

	float result = 0.;
	for (int i = 0; i < aWidth; i++) {
		result += a[aWidth * row + i] * b[bWidth * i + column];
	}

	out[bWidth * row + column] = result;
}

kernel void multiplyMatrixTile(global float* out, global float *a, global float *b, int bWidth, int aWidth, __local float *aBuffer, __local float *bBuffer)
{
	int column = get_local_id(0);
	int row = get_local_id(1);

	int blockColumn = get_group_id(0);
	int blockRow = get_group_id(1);

	int blockSize = get_local_size(0);

	int blockCount = (aWidth + blockSize - 1) / blockSize;

	float value = 0.;
	for (int i = 0; i < blockCount; i++) {
		int aGlobalColumn = i * blockSize + column;
		int aGlobalRow = blockRow * blockSize + row;
		aBuffer[row * blockSize + column] = a[aGlobalRow * aWidth + aGlobalColumn];

		int bGlobalColumn = blockColumn * blockSize + column;
		int bGlobalRow = i * blockSize + row;
		bBuffer[row * blockSize + column] = b[bGlobalRow * bWidth + bGlobalColumn];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int el = 0; el < blockSize; el++) {
			value += aBuffer[row * blockSize + el] * bBuffer[el * blockSize + column];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	int outRow = blockRow * blockSize + row;
	int outColumn = blockColumn * blockSize + column;

	out[bWidth * outRow + outColumn] = value;
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

void runKernel(cl_device_type deviceType, const char *kernelName, const Matrix& etalon, Matrix &out, const Matrix &a, const Matrix &b)
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

	cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, a.width * a.height * sizeof(float), NULL, &err);
	checkError(err, "clCreateBuffer d_a");

	cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, b.width * b.height * sizeof(float), NULL, &err);
	checkError(err, "clCreateBuffer d_b");

	cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out.width * out.height * sizeof(float), NULL, &err);
	checkError(err, "clCreateBuffer d_b");

	clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, a.width * a.height * sizeof(float), a.elements, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, b.width * b.height * sizeof(float), b.elements, 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(d_out), &d_out);
	checkError(err, "clSetKernelArg 0");

	err = clSetKernelArg(kernel, 1, sizeof(d_a), &d_a);
	checkError(err, "clSetKernelArg 1");

	err = clSetKernelArg(kernel, 2, sizeof(d_b), &d_b);
	checkError(err, "clSetKernelArg 2");

	err = clSetKernelArg(kernel, 3, sizeof(b.width), &b.width);
	checkError(err, "clSetKernelArg 3");

	err = clSetKernelArg(kernel, 4, sizeof(a.width), &a.width);
	checkError(err, "clSetKernelArg 4");

	cl_event event;

	const size_t blockSize = 16;
	size_t localWorkSize[2] = { blockSize, blockSize };
	size_t globalWorkSize[2] = {
		(out.width + blockSize - 1) / blockSize * blockSize,
		(out.height + blockSize - 1) / blockSize * blockSize
	};
	size_t localBufferSize = blockSize * blockSize * sizeof(float);

	err = clSetKernelArg(kernel, 5, localBufferSize, NULL);
	checkError(err, "clSetKernelArg 5");
	err = clSetKernelArg(kernel, 6, localBufferSize, NULL);
	checkError(err, "clSetKernelArg 6");

	clEnqueueNDRangeKernel(queue,
						   kernel,
						   2,
						   NULL,
						   globalWorkSize,
						   localWorkSize,
						   0,
						   NULL,
						   &event);

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
						out.width * out.height * sizeof(float),
						out.elements,
						0,
						0,
						NULL);
	checkError(err, "clEnqueueReadBuffer");

	cl_ulong elapsedTime = endTime - startTime;

	std::cout << deviceTypeStr(deviceType) << " kernel profile info: time takes " << elapsedTime << std::endl;

	bool theSameAnswer = 0 == std::memcmp(etalon.elements, out.elements, out.height * out.width * sizeof(float));

	std::cout << deviceTypeStr(deviceType) << " kernel gives and CPU loop the same result: " << theSameAnswer << std::endl;
	const std::string printValueMessage = std::string(deviceTypeStr(deviceType)) +" kernel first values";
	printFirstValues(printValueMessage.c_str(), out.elements, 10);

	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_out);
}

int main() {
	const size_t matrixSize = 1024;
	const size_t width = matrixSize;
	const size_t height = matrixSize;

	Matrix a(width, height);
	Matrix b(width, height);
	Matrix out(width, height, 0);
	Matrix outCpu(width, height, 0);
	Matrix outGpu(width, height, 0);

	runLoop(out, a, b);
	runKernel(CL_DEVICE_TYPE_CPU, "multiplyMatrix", out, outCpu, a, b);
	runKernel(CL_DEVICE_TYPE_GPU, "multiplyMatrixTile", out, outGpu, a, b);

	return 0;
}
