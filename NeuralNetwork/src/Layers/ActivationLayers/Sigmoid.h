#pragma once
#include "../../Interfaces/ILayer.h"

class Sigmoid : public ILayer
{
private:
	Eigen::MatrixXd outputs;

public:
	Eigen::MatrixXd Forward(Eigen::MatrixXd inputs) override;
	Eigen::MatrixXd Backward(Eigen::MatrixXd outputGradient, double learningRate) override;
};
