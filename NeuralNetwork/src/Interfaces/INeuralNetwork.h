#pragma once
#include <Eigen/Dense>

class INeuralNetwork
{
public:
	virtual Eigen::MatrixXd Prediction(Eigen::MatrixXd inputs) = 0;
	virtual void Fit(Eigen::MatrixXd inputs, Eigen::MatrixXd target, double learningRate, int epoch) = 0;
};