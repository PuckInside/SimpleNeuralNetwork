#pragma once
#include <Eigen/Dense>

class ILoss
{
public:
	virtual double Loss(Eigen::MatrixXd predict, Eigen::MatrixXd target) = 0;
	virtual Eigen::MatrixXd LossPrime(Eigen::MatrixXd predict, Eigen::MatrixXd target) = 0;
};