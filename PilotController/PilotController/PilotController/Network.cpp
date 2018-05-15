#include "Network.h"



Network::Network()
{
}


Network::~Network()
{
}

void Network::createNetwork(const DimensionNetwork & dimension)
{
	dim = dimension;
	n_sig = 0; n_neu = 0;
	neu = (Neurone**) new (Neurone*)[dim.layer];
	for (int i = 0; i < dim.layer; i++) {
		neu[i] = new Neurone[dim.size[i]];
		n_neu += dim.size[i];
		if (i > 0) {
			n_sig += dim.size[i] * dim.size[i-1];
		}
	}
	sig = new NetMath::Sigmoid[n_sig];
	int count = 0;
	for (int i = 0; i < dim.layer; i++) {
		neu[i] = new Neurone[dim.size[i]];
		n_neu += dim.size[i];
		if (i > 0) {
			n_sig += dim.size[i] * dim.size[i - 1];
		}
	}
}

