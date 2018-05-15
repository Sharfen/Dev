#include "Axone.h"



Axone::Axone()
{
}


Axone::~Axone()
{
}

CUDA_CALLABLE_MEMBER void Axone::getFromNeurone()
{
	sig->set(n_up->getValue());
}

CUDA_CALLABLE_MEMBER void Axone::setSigmoid(NetMath::Sigmoid * s)
{
	sig = s;
}

CUDA_CALLABLE_MEMBER void Axone::setNeurone(Neurone * n)
{
	n_up = n;
}
