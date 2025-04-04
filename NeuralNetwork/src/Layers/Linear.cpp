#include "Linear.h"

Linear::Linear(int inputSize, int outputSize)
{
	if (inputSize <= 0)
		throw std::exception();
	if (outputSize <= 0)
		throw std::exception();

	this->weights = Eigen::MatrixXd::Random(outputSize, inputSize);
	this->bias = Eigen::VectorXd::Random(outputSize);
}

Eigen::MatrixXd Linear::Forward(Eigen::MatrixXd inputs)
{
	this->inputs = inputs;
	int batchSize = inputs.cols();

	Eigen::MatrixXd predict = this->weights * this->inputs + this->bias.replicate(1, batchSize);
	return predict;
}


Eigen::MatrixXd Linear::Backward(Eigen::MatrixXd outputGradient, double learningRate)
{
	this->weights -= outputGradient * this->inputs.transpose() * 0.01;
	this->bias -= outputGradient.rowwise().sum() * 0.01;

	Eigen::MatrixXd inputGradient = this->weights.transpose() * outputGradient;
	return inputGradient;
}
