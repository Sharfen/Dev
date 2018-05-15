#pragma once
#include "Axone.h"
#include <vector>

/* Network homds the communication between host and device
	Parallelised resources are instanciated on the device while
	global resources are on the host
*/

struct DimensionNetwork {
	int layer;
	std::vector<int> size;
	void addLayer(const int &s) {
		size.push_back(s);
		layer = size.size();
	}
};

class Network
{
public:
	Network();
	~Network();

	void createNetwork(const DimensionNetwork &dimension);

private:
	Neurone **neu;
	NetMath::Sigmoid *sig;
	Axone *axone;
	DimensionNetwork dim;
	int n_sig, n_neu;

	Axone * d_axone;
	NetMath::Sigmoid *d_sig;
	Neurone *d_neu;
};

