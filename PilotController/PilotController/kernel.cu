

#include "Network.h"
#include <iostream>
#include <Windows.h>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


void testSimpleSigmoid() {
	NetMath::Sigmoid *sig = new NetMath::Sigmoid(1.0, 10.0, -10.0);
	NetMath::Sigmoid target(1.0, 0.0, -1.0);
	const int tabSize = 11;
	nm_float x[tabSize] = { -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0 };
	nm_float t[tabSize];
	for (int i = 0; i < tabSize; i++) {
		t[i] = target.getSigmoid(x[i]);
	}
	std::cout << NetMath::Sigmoid::distance(*sig, target) << std::endl;
	for (int i = 0; i < 1; i++) {

		NetMath::MethodMultiGradient(sig, 1, x, t, tabSize, 0.0001, 0.0000001, 0.1);
		std::cout << NetMath::Sigmoid::distance(*sig, target) << std::endl;
		sig->getTheta();
		for (int i = 0; i < tabSize; i++) {
			std::cout << target.getSigmoid(x[i]) << " " << sig->getSigmoid(x[i]) << std::endl;
		}
	}
	delete sig;
}

void testNetwork() {
	Network net;
	net.addDimension(-5);
	net.addDimension(4);
	net.addDimension(6);
	net.createNetwork();
	for (int i = 0; i < net.getInputSize(); i++)
		net.input[i] = 0;
	net.test();
	for (int i = 0; i < net.getOutputSize(); i++)
		std::cout << net.output[i] << " ";
	std::cout << std::endl;
	system("pause");
}

void testParallelSigmoid() {

	NetMath::Sigmoid *sig(NULL);
	NetMath::Sigmoid target(1.0, 0.0, -1.0);
	const int tabSize = 13;
	nm_float x[tabSize] = { -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0 };
	nm_float t[tabSize];
	const int nsig = 6;
	sig = new NetMath::Sigmoid[nsig];
	for (int i = 0; i < nsig; i++) {
		sig[i].setTheta(1.0, 2.0 *i / (nsig - 1) - 1.0, 1.0 * i / (nsig - 1) - 0.5);
		sig[i].getTheta();
	}
	for (int i = 0; i < tabSize; i++) {
		t[i] = x[i] * x[i];
	}
	for (int i = 0; i < 1; i++) {

		NetMath::MethodMultiGradient(sig, nsig, x, t, tabSize, 0.0001, 0.0000001, 0.5);
		//std::cout << NetMath::Sigmoid::distance(*sig, target) << std::endl;
		//sig->getTheta();
		for (int i = 0; i < tabSize; i++) {
			NetMath::Set(sig, nsig, x[i]);
			std::cout << t[i] << " " << NetMath::Value(sig, nsig) << std::endl;
		}
		for (int i = 0; i < nsig; i++) {
			sig[i].getTheta();
		}
	}
	nm_float sharp_x(-3.0), err(0.0);
	for (int i = 0; i < 1000; i++) {
		for (int i = 0; i < tabSize; i++) {
			NetMath::Set(sig, nsig, sharp_x);
			err += fabsf(NetMath::Value(sig, nsig) - sharp_x*sharp_x);
		}
		sharp_x += 6.0 / 1000;
	}
	err /= 1000;
	std::cout << "err avg : " << err << std::endl;
	delete[] sig;
	system("pause");
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	//testParallelSigmoid();
	testNetwork();
	system("pause");

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


	system("pause");

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size, 1>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
