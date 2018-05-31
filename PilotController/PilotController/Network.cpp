#include "Network.h"
#include "Kernels.h"


Network::Network() : locked(false),
d_dimension(NULL), 
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

	dimension.size = size.size();
	dimension.in_size = size[0];
	dimension.out_size = size[size.size() - 1];
	dimension.cur_layer = 0;

	cudaStatus = cudaMalloc((void**)&d_dimension, sizeof(Dimension));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate dimension");
		return cudaStatus;
	}

	int s = size.size();

	cudaStatus = cudaMalloc((void**)&dimension.value, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate value");
		return cudaStatus;
	}

	int *v = new int[s];
	for (int i = 0; i < s; i++)
		v[i] = size[i];

	cudaStatus = cudaMemcpy(dimension.value, &v, s*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set value");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.pitch, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate pitch");
		return cudaStatus;
	}

	v[0] = 0;
	for (int i = 1; i < s; i++)
		v[i] = v[i-1] + size[i-1];

	cudaStatus = cudaMemcpy(dimension.pitch, &v, s * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.sig_value, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate sig_pitch");
		return cudaStatus;
	}

	for (int i = 0; i < s-1; i++)
		v[i] = size[i] * size[i+1];

	cudaStatus = cudaMemcpy(dimension.sig_value, &v, s * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set sig_pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dimension.sig_pitch, s * sizeof(int));
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to allocate sig_pitch");
		return cudaStatus;
	}

	v[0] = 0;
	for (int i = 1; i < s; i++)
		v[i] = v[i - 1] + size[i - 1] * size[i];

	cudaStatus = cudaMemcpy(dimension.sig_pitch, &v, s * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set sig_pitch");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_dimension, &dimension, sizeof(Dimension), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "failed to set dimension");
		return cudaStatus;
	}

	return CUDA_SUCCESS;
}

CUDA_ERROR Network::deleteDimension()
{
	if (d_dimension != NULL) {
		cudaFree(dimension.value);
		cudaFree(dimension.pitch);
		cudaFree(dimension.sig_value);
		cudaFree(dimension.sig_pitch);
		cudaFree(d_dimension);
	}
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
	int s(0);

	for (int i = 1; i < size.size(); i++)
		s += size[i - 1] * size[i];

	return cudaMalloc((void**)&d_neu, s * sizeof(nm_float));
}

CUDA_ERROR Network::deleteSigmoids()
{
	return cudaFree(d_sig);
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
	for (int i = 0; i < dimension.size; i++) {
		setLayer(i);
		if (i != 0) {
			kernel_propagate_neu<<<1, dimension.value[i], 1 >>>(d_neu, d_sig, d_dimension);
		} if (i != dimension.size - 1){
			kernel_propagate_sig<<<1, dimension.sig_value[i], 1 >>>(d_neu, d_sig, d_dimension);
		}
	}
}

CUDA_ERROR Network::setLayer(const int & layer)
{
	int l(layer);
	if (layer < 0 || layer >= size.size()) {
		Log(ERROR, "no layer nb ", layer);
		l = 0;
	}
	cudaError_t cudaStatus = cudaMemcpy(&d_dimension->cur_layer, &l, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != CUDA_SUCCESS)
		Log(ERROR, "failed to set layer", l);
	return cudaStatus;
}

