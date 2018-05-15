#pragma once
#include "Neurone.h"

class Axone
{
public:
	Axone();
	virtual ~Axone();

	CUDA_CALLABLE_MEMBER void getFromNeurone();
	CUDA_CALLABLE_MEMBER void setSigmoid(NetMath::Sigmoid *s);
	CUDA_CALLABLE_MEMBER void setNeurone(Neurone *n);

public:
	NetMath::Sigmoid *sig;
	Neurone *n_up;
	
};

