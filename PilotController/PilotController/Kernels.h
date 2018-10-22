#pragma once
#include "Network.h"
#include <string>


__global__ void kernel_propagate_sig(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids,
	Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_pitch);


__global__ void kernel_propagate_neu(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids,
	Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_value);

namespace KernelCallers {
	// Tools
	CUDA_ERROR check_exec(const std::string &s);

	// Network functions callers


	CUDA_ERROR propagate_sig(nm_float * d_neu, NetMath::Sigmoid * d_sig, uint dimension,
		Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_pitch);

	CUDA_ERROR propagate_neu(nm_float * d_neu, NetMath::Sigmoid * d_sig, uint dimension,
		Network::Dimension::Parameter * param, int *value, int *pitch, int *sig_value);
	CUDA_ERROR set_layer(Network::Dimension * d_dimension, const int &layer);
	
}

namespace KernelChecker {
	CUDA_ERROR propagate_sig(Network::Dimension *dimension, const uint &poolsize);
}
