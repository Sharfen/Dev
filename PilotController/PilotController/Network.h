#pragma once
#include "Axone.h"
#include <vector>
#include <iostream>

/* Network homds the communication between host and device
	Parallelised resources are instanciated on the device while
	global resources are on the host
*/



class Network
{
	struct Dimension {
		int size(0), 	// size of arrays
		*value, 		// individual size of layers
		*pitch;			// cumulated size of layers ( pitch[k] = S_0^k-1 value[i] )
		*sig_pitch		// cumulated pitch for sigmoids ( sig_pitch[k] = S_0^k-1 value[i]*value[i+1] )
	};

public:
	Network();
	~Network();
	
	/* Calls
		preCheck
		createDimension
		createNeurones
		createSigmoids
	*/
	CUDA_ERROR createNetwork();
	
	/// Simply add a dimension
	int addDimension(const int &s);

private:

	CUDA_ERROR preCheck();
	CUDA_ERROR postCheck();
	
	CUDA_ERROR createDimension();
	CUDA_ERROR deleteDimension();
	
	CUDA_ERROR createNeurones();
	CUDA_ERROR deleteNeurones();
	
	CUDA_ERROR createSigmoids();
	CUDA_ERROR deleteSigmoids();
	
	CUDA_ERROR copyInput();
	CUDA_ERROR copyOutput();
	
	CUDA_ERROR propagate();
	
	CUDA_ERROR processGradient();
	
	CUDA_ERROR backPropagate();
	
	CUDA_ERROR saveStack();
	
	std::vector<int> size;
	
	bool locked;
	Dimension dimension, *d_dimension;
	NetMath::Sigmoid *d_sig;
	nm_float *d_neu;
	nm_float *input, *output;
	std::fstream executionStack, network;
};

