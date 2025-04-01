#pragma once
#include <Eigen/Dense>

class ILayer
{
public:
	virtual Eigen::MatrixXd Forward(Eigen::MatrixXd inputs) = 0;
	virtual Eigen::MatrixXd Backward(Eigen::MatrixXd outputGradient, double learningRate) = 0;
};