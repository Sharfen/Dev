#pragma once
#include "NetMath.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>

/* Network homds the communication between host and device
	Parallelised resources are instanciated on the device while
	global resources are on the host
*/



class Network
{
public:
	struct Dimension {
		struct Parameter {
			int size,			// size of arrays
				sig_size,
				in_size,
				out_size,
				cur_layer;
		};
		Parameter param, *d_param;
		int	*value, 		// individual size of layers
			*pitch,			// cumulated size of layers ( pitch[k] = S_0^k-1 value[i] )
			*sig_value,		// quantity of sigs linking the layers
			*sig_pitch,		// cumulated pitch for sigmoids ( sig_pitch[k] = S_0^k-1 value[i]*value[i+1] )
			*d_value,
			*d_pitch,
			*d_sig_value,
			*d_sig_pitch;
	};

	Network();
	~Network();
	
	/* Calls
		preCheck
		createDimension
		createNeurones
		createSigmoids
	*/
	CUDA_ERROR createNetwork();
	CUDA_ERROR deleteNetwork();
	
	/// Simply add a dimension
	int addDimension(const int &s);

	// test functions
	CUDA_ERROR test();


	nm_float *input, *output;
	int getInputSize() const;
	int getOutputSize() const;

private:


	CUDA_ERROR preCheck();
	CUDA_ERROR postCheck();
	
	CUDA_ERROR createDimension();
	CUDA_ERROR deleteDimension();
	
	CUDA_ERROR createNeurones();
	CUDA_ERROR deleteNeurones();
	
	CUDA_ERROR createSigmoids();
	CUDA_ERROR deleteSigmoids();

	CUDA_ERROR setupSigmoids();
	
	CUDA_ERROR copyInput();
	CUDA_ERROR copyOutput();
	
	CUDA_ERROR propagate();
	
	CUDA_ERROR processGradient();
	
	CUDA_ERROR backPropagate();
	
	CUDA_ERROR saveStack();

	CUDA_ERROR setLayer(const int &layer);

	// Kernels

	
	std::vector<int> size;
	
	bool locked;
	Dimension dimension;
	NetMath::Sigmoid *d_sig;
	nm_float *d_neu;
	int out_layer_offset, input_size, output_size;
	std::fstream executionStack, network;
};

