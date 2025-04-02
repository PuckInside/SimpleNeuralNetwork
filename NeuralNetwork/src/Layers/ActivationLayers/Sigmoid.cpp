#include "Sigmoid.h"

Eigen::MatrixXd Sigmoid::Forward(Eigen::MatrixXd inputs)
{
	this->outputs = 1.0 / (1.0 + (-inputs.array()).exp());
	return outputs;
}

Eigen::MatrixXd Sigmoid::Backward(Eigen::MatrixXd outputGradient, double learningRate)
{
	Eigen::MatrixXd activationPrime = this->outputs.array() * (1 - this->outputs.array());
	Eigen::MatrixXd inputGradient = outputGradient.array() * activationPrime.array();
	return inputGradient;
}
