#include "Kernels.h"

__global__ void kernel_propagate_sig(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids, 
	Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_pitch)
{
	int neu_src = threadIdx.x / value[param->cur_layer] + pitch[param->cur_layer],
		sig = sig_pitch[param->cur_layer] + threadIdx.x;
		
	d_sigmoids[sig].set(d_neurones[neu_src]);
	
}

__global__ void kernel_propagate_neu(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids,
	Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_value)
{
	int neu = threadIdx.x + pitch[param->cur_layer];
	d_neurones[neu] = 0.0;
	for (int i = threadIdx.x; i < sig_value[param->cur_layer - 1]; i += value[param->cur_layer - 1])
		d_neurones[neu] += d_sigmoids[i]();
}

CUDA_ERROR KernelCallers::check_exec(const std::string &s)
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "kernel function ", s, " failed : ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(ERROR, "function ", s, " failed to synchronize ", cudaStatus);
	}
	return cudaStatus;
}

CUDA_ERROR KernelCallers::propagate_sig(nm_float * d_neu, NetMath::Sigmoid * d_sig, uint dimension,
	Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_pitch)
{
	kernel_propagate_sig<<<1, dimension, 1 >>>(d_neu, d_sig, param, value, pitch, sig_pitch);
	return check_exec("propagate_sig");
}

CUDA_ERROR KernelCallers::propagate_neu(nm_float * d_neu, NetMath::Sigmoid * d_sig, uint dimension,
	Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_value)
{
	kernel_propagate_neu<<<1, dimension, 1 >>>(d_neu, d_sig, param, value, pitch, sig_value);
	return check_exec("propagate_neu");
}

CUDA_ERROR KernelChecker::propagate_sig(Network::Dimension *d_dim, const uint & poolsize)
{
	for(int i=0; i<poolsize; i++)
	std::cout << i / d_dim->value[d_dim->param.cur_layer] + d_dim->pitch[d_dim->param.cur_layer]
		<< " " << d_dim->sig_pitch[d_dim->param.cur_layer] + i << std::endl;
	return CUDA_SUCCESS;
}
