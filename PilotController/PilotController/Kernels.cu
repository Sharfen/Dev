#include "Kernels.h"


__global__ void kernel_propagate_sig(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids, Network::Dimension * d_dim)
{
	int neu_src = threadIdx.x / d_dim->value[d_dim->cur_layer] + d_dim->pitch[d_dim->cur_layer],
		sig = d_dim->sig_pitch[d_dim->cur_layer] + threadIdx.x;
	d_sigmoids[sig].set(d_neurones[neu_src]);
}

__global__ void kernel_propagate_neu(nm_float * d_neurones, NetMath::Sigmoid * d_sigmoids, Network::Dimension * d_dim)
{
	int neu = threadIdx.x + d_dim->pitch[d_dim->cur_layer];
	d_neurones[neu] = 0.0;
	for (int i = threadIdx.x; i < d_dim->sig_value[d_dim->cur_layer - 1]; i += d_dim->value[d_dim->cur_layer - 1])
		d_neurones[neu] += d_sigmoids[i]();
}