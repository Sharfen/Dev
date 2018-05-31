#include "Neurone.h"



Neurone::Neurone()
{
}


Neurone::~Neurone()
{
}

CUDA_CALLABLE_MEMBER void Neurone::setValueFromSigmoid()
{
	next_value = NetMath::Value(feeding_sig, feed_size);
}

CUDA_CALLABLE_MEMBER void Neurone::updateValue()
{
	last_value = next_value;
}

CUDA_CALLABLE_MEMBER nm_float Neurone::getValue() const
{
	return last_value;
}
