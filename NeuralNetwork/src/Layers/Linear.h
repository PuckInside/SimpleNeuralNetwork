#pragma once
#include "../Interfaces/ILayer.h"

class Linear : public ILayer
{
private:
	Eigen::MatrixXd weights;
	Eigen::VectorXd bias;
	Eigen::MatrixXd inputs;

public:
	Linear(int inputSize, int outputSize);

	Eigen::MatrixXd Forward(Eigen::MatrixXd inputs) override;
	Eigen::MatrixXd Backward(Eigen::MatrixXd outputGradient, double learningRate) override;
};

