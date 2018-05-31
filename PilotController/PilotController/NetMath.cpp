#include "NetMath.h"


CUDA_CALLABLE_MEMBER NetMath::Sigmoid::Sigmoid()
{
}

CUDA_CALLABLE_MEMBER NetMath::Sigmoid::Sigmoid(const nm_float & alpha, const nm_float & beta, const nm_float & gamma)
{
	theta = { alpha, beta, gamma };
}

CUDA_CALLABLE_MEMBER NetMath::Sigmoid::~Sigmoid()
{
}

void NetMath::Sigmoid::setTheta(const nm_float & alpha, const nm_float & beta, const nm_float & gamma)
{
	theta = { alpha, beta, gamma };
}

void NetMath::Sigmoid::getTheta() const
{
	std::cout << theta.alpha << " " << theta.beta << " " << theta.gamma << std::endl;
}

nm_float NetMath::Sigmoid::getSigmoid(const nm_float & x)
{
	set(x);
	return last_sig;
}

CUDA_CALLABLE_MEMBER nm_float NetMath::Sigmoid::operator()() const
{
	return last_sig;
}

CUDA_CALLABLE_MEMBER void NetMath::Sigmoid::set(const nm_float & x)
{
	last_x = x;
	recomputeSigmoid();
}

void NetMath::Sigmoid::recomputeSigmoid()
{
	last_sig = theta.alpha / (1.0 + exp(-1.0*theta.gamma * (last_x - theta.beta)));
}

void NetMath::Sigmoid::grad()
{
	grad_theta = { grad_alpha(), grad_beta(), grad_gamma() };
}

void NetMath::Sigmoid::powGrad(const int & p)
{
	grad_theta ^= p;
}

void NetMath::Sigmoid::resetMultiGrad()
{
	multi_grad = { 0.0, 0.0, 0.0 };
}

void NetMath::Sigmoid::addMultiGrad(const nm_float & quantity)
{
	multi_grad += grad_theta * quantity;
}

void NetMath::Sigmoid::moveMultiGrad()
{
	theta += multi_grad;
}

nm_float NetMath::Sigmoid::getMultiGrad() const
{
	return multi_grad.abssum();
}

nm_float NetMath::Sigmoid::getGrad() const
{
	return grad_theta.sum();
}

nm_float NetMath::Sigmoid::getAbsGrad() const
{
	return grad_theta.abssum();
}

nm_float NetMath::Sigmoid::getX() const
{
	return last_x;
}

void NetMath::Sigmoid::moveGrad(const nm_float & quantity)
{
	theta+=grad_theta*quantity;
}

nm_float NetMath::Sigmoid::distance(const Sigmoid & s1, const Sigmoid & s2)
{
	Theta theta(s1.theta);
	theta += s2.theta*(-1.0);
	return theta.abssum();;
}

nm_float NetMath::Sigmoid::grad_alpha() const
{
	return last_sig /theta.alpha;
	//return 0.0;
}

nm_float NetMath::Sigmoid::grad_beta() const
{
	return - theta.gamma / theta.alpha / theta.alpha *last_sig*(theta.alpha - last_sig);
	//return 0.0;
}

nm_float NetMath::Sigmoid::grad_gamma() const
{
	return (last_x-theta.beta)/ theta.alpha/ theta.alpha *last_sig*(theta.alpha - last_sig);
}

nm_float NetMath::Sigmoid::Theta::sum() const
{
	return alpha+beta+gamma;
}

nm_float NetMath::Sigmoid::Theta::abssum() const
{
	return abs(alpha) + abs(beta) + abs(gamma);
}

void NetMath::Sigmoid::Theta::operator+=(const Theta & t)
{
	alpha += t.alpha; beta += t.beta; gamma += t.gamma;
}

void NetMath::Sigmoid::Theta::operator^=(const int & p)
{
	alpha = pow(alpha, p);
	beta = pow(beta, p);
	gamma = pow(gamma, p);
}

NetMath::Sigmoid::Theta NetMath::Sigmoid::Theta::operator*(const nm_float & x) const
{
	return { x*alpha, x*beta, x*gamma };
}
