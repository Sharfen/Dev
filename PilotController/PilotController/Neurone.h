#pragma once

#include "NetMath.h"

// Very basis of a neurone comportment
class Neurone
{
public:
	CUDA_CALLABLE_MEMBER Neurone();
	CUDA_CALLABLE_MEMBER virtual ~Neurone();

	CUDA_CALLABLE_MEMBER void setValueFromSigmoid();
	CUDA_CALLABLE_MEMBER void updateValue();
	CUDA_CALLABLE_MEMBER nm_float getValue() const;

private:
	nm_float next_value, last_value;
	NetMath::Sigmoid *feeding_sig;
	int feed_size;
};

// Pilot neurone
// Uses the controller value for filtering its values

