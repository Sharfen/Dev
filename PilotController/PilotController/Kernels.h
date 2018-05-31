#pragma once
#include "Network.h"

__global__ void kernel_propagate_sig(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids, Network::Dimension * d_dim);

__global__ void kernel_propagate_neu(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids, Network::Dimension * d_dim);