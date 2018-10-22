#include "Network.h"
#include "Kernels.h"


Network::Network() : locked(false),
d_sig(NULL),
d_neu(NULL),
input(NULL), 
output(NULL)
{
}


Network::~Network()
{
	deleteNetwork();
}

CUDA_ERROR Network::createNetwork()
{
	cudaError_t cudaStatus;
	if (locked) {
		Log(ERROR, "network locked");
		return (CUDA_ERROR)-1;
	}
	if (size.size() == 0) {
		Log(ERROR, "network size is 0");
		return (CUDA_ERROR)-2;
	}

	cudaStatus = preCheck();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "unable to set cuda device");
		return cudaStatus;
	}

	cudaStatus = createDimension();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "unable to create dimension data structure");
		return cudaStatus;
	}

	cudaStatus = createNeurones();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "unable to allocate neurones");
		return cudaStatus;
	}

	cudaStatus = createSigmoids();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "unable to allocate sigmoids");
		return cudaStatus;
	}

	cudaStatus = setupSigmoids();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "unable to set sigmoids");
		return cudaStatus;
	}

	Log(INFO, "network created");
	return cudaStatus;
}

CUDA_ERROR Network::deleteNetwork()
{
	cudaError_t cudaStatus;
	cudaStatus = postCheck();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed on postcheck");
	}

	cudaStatus = deleteDimension();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to delete dimension");
	}

	cudaStatus = deleteNeurones();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to delete neurones");
	}

	cudaStatus = deleteSigmoids();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to delete sigmoids");
	}

	Log(INFO, "network deleted");
	return CUDA_SUCCESS;
}

int Network::addDimension(const int & s)
{
	if (s <= 0) {
		Log(ERROR, "cannot add dimension of size ", s);
		return -1;
	}
	size.push_back(s);
	Log(INFO, "added dimension ", s);
	return 0;
}

CUDA_ERROR Network::test()
{
	copyInput();
	propagate();
	
	return copyOutput();
}

int Network::getInputSize() const
{
	return input_size;
}

int Network::getOutputSize() const
{
	return output_size;
}

CUDA_ERROR Network::preCheck()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set device");
		return cudaStatus;
	}
	Log(INFO, "device setted");
	return CUDA_SUCCESS;
}

CUDA_ERROR Network::postCheck()
{
	return CUDA_SUCCESS;
}

CUDA_ERROR Network::createDimension()
{
	cudaError_t cudaStatus;

	// Set parameters on host

	dimension.param.size = size.size();
	dimension.param.sig_size = 0;
	for (int i = 1; i < dimension.param.size; i++)
		dimension.param.sig_size += size[i - 1] * size[i];
	dimension.param.in_size = size[0];
	dimension.param.out_size = size[size.size() - 1];
	dimension.param.cur_layer = 0;

	// Set layers parameters on host

	int s = dimension.param.size;

	dimension.value = new int[dimension.param.size];
	for (int i = 0; i < s; i++)
		dimension.value[i] = size[i];

	dimension.pitch = new int[dimension.param.size];
	dimension.pitch[0] = 0;
	for (int i = 1; i < s; i++)
		dimension.pitch[i] = dimension.pitch[i - 1] + size[i - 1];

	dimension.sig_value = new int[dimension.param.size];
	for (int i = 0; i < s - 1; i++)
		dimension.sig_value[i] = size[i] * size[i + 1];

	dimension.sig_pitch = new int[dimension.param.size];
	dimension.sig_pitch[0] = 0;
	for (int i = 1; i < s; i++)
		dimension.sig_pitch[i] = dimension.sig_pitch[i - 1] + size[i - 1] * size[i];

	// Allocate parameters & layer parameters on device

	cudaStatus = cudaMalloc((void**)&dimension.d_param, sizeof(Dimension::Parameter));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate parameters");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.d_value, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate value");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.d_pitch, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.d_sig_value, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate sig_pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.d_sig_pitch, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate sig_pitch");
		return cudaStatus;
	}

	// Copy from host to device

	cudaStatus = cudaMemcpy(dimension.d_param, &dimension.param, sizeof(Dimension::Parameter), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set parameters");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dimension.d_value, dimension.value, s*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set value");
		return cudaStatus;
	}


	cudaStatus = cudaMemcpy(dimension.d_pitch, dimension.pitch, s * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dimension.d_sig_value, dimension.sig_value, s * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set sig_pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dimension.d_sig_pitch, dimension.sig_pitch, s * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set sig_pitch");
		return cudaStatus;
	}

	//KernelChecker::checkDimension(d_dimension);

	Log(INFO, "dimension created");

	return CUDA_SUCCESS;
}

CUDA_ERROR Network::deleteDimension()
{
		cudaFree(dimension.d_value);
		cudaFree(dimension.d_pitch);
		cudaFree(dimension.d_sig_value);
		cudaFree(dimension.d_sig_pitch);
		cudaFree(dimension.d_param);

	delete[] dimension.value;
	delete[] dimension.pitch;
	delete[] dimension.sig_value;
	delete[] dimension.sig_pitch;

	return CUDA_SUCCESS;
}

CUDA_ERROR Network::createNeurones()
{
	int s(0);
	input_size = size[0];
	output_size = size[size.size() - 1];

	input = new nm_float[input_size];
	output = new nm_float[output_size];
	for (auto it = size.begin(); it != size.end(); ++it)
		s += *it;
	out_layer_offset = s - output_size;

	Log(INFO, "allocate neurone board of ", s);

	return cudaMalloc((void**)&d_neu, s * sizeof(nm_float));
}

CUDA_ERROR Network::deleteNeurones()
{
	delete[] input;
	delete[] output;
	return cudaFree(d_neu);
}

CUDA_ERROR Network::createSigmoids()
{
	Log(INFO, "allocate sigmoid board of ", dimension.param.sig_size);

	return cudaMalloc((void**)&d_sig, dimension.param.sig_size * sizeof(NetMath::Sigmoid));
}

CUDA_ERROR Network::deleteSigmoids()
{
	return cudaFree(d_sig);
}

CUDA_ERROR Network::setupSigmoids()
{
	NetMath::Sigmoid *sigmoids = new NetMath::Sigmoid[dimension.param.sig_size];
	nm_float min_b(-2.0), max_b(2.0), min_g(-2.0), max_g(2.0), b(min_b), g(min_g);
	for (int i = 0; i < dimension.param.sig_size; i++) {
		sigmoids[i].setupSigmoid();
		sigmoids[i].setTheta(1.0, b, g);
		b += (max_b - min_b) / dimension.param.sig_size;
		g += (max_g - min_g) / dimension.param.sig_size;
	}
	Log(INFO, "copying sigmoids board of ", dimension.param.sig_size);
	CUDA_ERROR cudaStatus = cudaMemcpy(d_sig, sigmoids, dimension.param.sig_size * sizeof(NetMath::Sigmoid), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set sigmoids");
		return cudaStatus;
	}
	delete[] sigmoids;
	return cudaStatus;
}

CUDA_ERROR Network::copyInput()
{
	return cudaMemcpy(d_neu, input, input_size * sizeof(nm_float), cudaMemcpyHostToDevice);
}

CUDA_ERROR Network::copyOutput()
{
	return cudaMemcpy(output, d_neu+out_layer_offset, output_size * sizeof(nm_float), cudaMemcpyDeviceToHost);
}

CUDA_ERROR Network::propagate()
{
	for (int i = 0; i < dimension.param.size; i++) {
		setLayer(i);
		if (i != 0) {
			KernelCallers::propagate_neu(d_neu, d_sig, dimension.value[dimension.param.cur_layer], dimension.d_param, dimension.d_value, dimension.d_pitch, dimension.d_sig_value);
		} 
		if (i != dimension.param.size - 1) {
			//KernelChecker::propagate_sig(&dimension, dimension.sig_value[dimension.cur_layer]);
			KernelCallers::propagate_sig(d_neu, d_sig, dimension.sig_value[dimension.param.cur_layer], dimension.d_param, dimension.d_value, dimension.d_pitch, dimension.d_sig_pitch);
		}
	}
	return CUDA_SUCCESS;
}

CUDA_ERROR Network::setLayer(const int & layer)
{
	int l(layer);
	if (layer < 0 || layer >= size.size()) {
		Log(ERROR, "no layer nb ", layer);
		l = 0;
	}
	dimension.param.cur_layer = l;
	cudaError_t cudaStatus = cudaMemcpy(dimension.d_param, &dimension.param, sizeof(Dimension::Parameter), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS)
		Log(ERROR, "failed to set layer ", l);
	return cudaStatus;
}

