#pragma once
#include <cmath>
#include <time.h>
#define nm_float double
#define nm_int int
#include <iostream>

namespace NetMath {
	class Sigmoid {
		/*Sigmoid s :
			theta={	alpha	: range
					beta	: seuil
					gamma	: pente
				}
			s(theta, x) = alpha / (1 + exp( gamma * (x - beta ) ) )
		*/
	public:
		struct Theta {
			nm_float alpha, beta, gamma;
			nm_float sum() const;
			nm_float abssum() const;
			void operator+=(const Theta &t);
			void operator^=(const int &p);
			Theta operator*(const nm_float &x) const;
		};

		Sigmoid();
		Sigmoid(const nm_float &alpha, const nm_float &beta, const nm_float &gamma);
		~Sigmoid();

		void getTheta() const;
		nm_float getSigmoid(const nm_float &x);
		nm_float operator()() const;
		void set(const nm_float &x);
		void recomputeSigmoid();
		void grad();
		void powGrad(const int & p);
		void resetMultiGrad();
		void addMultiGrad(const nm_float &quantity);
		void moveMultiGrad();
		nm_float getMultiGrad() const;
		nm_float getGrad() const;
		nm_float getAbsGrad() const;
		nm_float getX() const;
		void moveGrad(const nm_float &quantity);

		static nm_float distance(const Sigmoid &s1, const Sigmoid &s2);

	private:
		Theta theta, grad_theta, multi_grad;
		nm_float last_sig, last_x;
		nm_float grad_alpha() const;
		nm_float grad_beta() const;
		nm_float grad_gamma() const;
	};

	namespace {
		nm_float GradLengh(Sigmoid *sig, const nm_int &size) {
			nm_float quantity(0.0);
			for (nm_int i = 0; i < size; i++){
				sig[i].grad();
				quantity += sig[i].getAbsGrad();
			}
			return quantity;
		}

		nm_float Value(Sigmoid *sig, const nm_int &size) {
			nm_float value(0.0);
			for (nm_int i = 0; i < size; i++)
				value += sig[i]();
			return value;
		}

		void MoveGrad(Sigmoid *sig, const nm_int &size, const nm_float &quantity) {
			for (nm_int i = 0; i < size; i++) {
				sig[i].moveGrad(quantity);
				sig[i].recomputeSigmoid();
			}
		}	

		void AddMultiGrad(Sigmoid *sig, const nm_int &size, const nm_float &quantity) {
			for (nm_int i = 0; i < size; i++) {
				sig[i].addMultiGrad(quantity);
			}
		}

		void MoveMultiGrad(Sigmoid *sig, const nm_int &size) {
			for (nm_int i = 0; i < size; i++) {
				sig[i].moveMultiGrad();
			}
		}

		void Set(Sigmoid *sig, const nm_int &size, const nm_float &x) {
			for (nm_int i = 0; i < size; i++) {
				sig[i].set(x);
			}
		}

		nm_float MultiGradDistance(Sigmoid *sig, const nm_int &size, const nm_float *x, const nm_float *target, const int &size_sample) {
			nm_float distance(0.0);
			for (int i = 0; i<size_sample; i++) {
				Set(sig, size, x[i]);
				distance += abs(Value(sig, size) - target[i]);
			}
			return distance;
		}

		nm_float Steepness(Sigmoid *sig, const nm_int &size) {
			nm_float steepness(0.0);
			for (int i = 0; i < size; i++)
				steepness += sig[i].getMultiGrad();
			return steepness;
		}

		void PowGrad(Sigmoid *sig, const nm_int &size, const int &p) {
			for (int i = 0; i < size; i++)
				sig[i].powGrad(p);
		}
	}

	inline void MethodGradient(Sigmoid *sig, const nm_int &size, const nm_float &target,
								const nm_float &epsilon, const nm_float &speed) {
		nm_float quantity, lengh, last(Value(sig, size));
		while (abs(target - last) > epsilon) {
			lengh = GradLengh(sig, size);
			quantity = (target - last) / lengh * speed;
			MoveGrad(sig, size, quantity);
			last = Value(sig, size);
			std::cout << "\r    " << abs(target - last);
		} 
		std::cout << std::endl;
	}

	inline void MethodMultiGradient(Sigmoid *sig, const nm_int &size, const nm_float *x, const nm_float *target, const int &size_sample, 
									const nm_float &epsilon, const nm_float &grad_epsilon, const nm_float &speed) {
		nm_float quantity, lengh, distance(0.0), steepness(grad_epsilon+1.0);
		int count(0);
		distance = MultiGradDistance(sig, size, x, target, size_sample);
		while (distance > epsilon && steepness > grad_epsilon) {
			for (int i = 0; i < size; i++)
				sig[i].resetMultiGrad();
			distance = 0.0;
			lengh = 0.0;
			for (int i = 0; i < size_sample; i++) {
				Set(sig, size, x[i]);
				lengh = GradLengh(sig, size);
				//std::cout << lengh << " " << target[i] - Value(sig, size) << std::endl;
				if(lengh > 0.0) {
					quantity = (target[i] - Value(sig, size)) / lengh * speed;
					//std::cout << quantity << std::endl;
					AddMultiGrad(sig, size, quantity);
				}
			}
			MoveMultiGrad(sig, size);
			distance = MultiGradDistance(sig, size, x, target, size_sample);
			steepness = Steepness(sig, size);
			if(count++%10000 == 0)
				std::cout << "\r    " << count << " " << distance << " " << steepness << "    ";
		}
		std::cout << std::endl;
	}
};

