#include "Linear.h"

Linear::Linear(int inputSize, int outputSize)
{
	this->weights = Eigen::MatrixXd::Random(outputSize, inputSize);
	this->bias = Eigen::VectorXd::Random(outputSize);
}

Eigen::MatrixXd Linear::Forward(Eigen::MatrixXd inputs)
{
	this->inputs = inputs;
	Eigen::MatrixXd predict = this->weights * inputs + this->bias;
	return predict;
}

Eigen::MatrixXd Linear::Backward(Eigen::MatrixXd outputGradient, double learningRate)
{
	this->weights -= outputGradient * this->inputs.transpose() * 0.01;
	this->bias -= outputGradient * 0.01;

	Eigen::MatrixXd inputGradient = this->weights.transpose() * outputGradient;
	return inputGradient;
}
