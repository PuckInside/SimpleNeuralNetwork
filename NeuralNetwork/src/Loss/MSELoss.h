#pragma once
#include "../Interfaces/ILoss.h"

class MSELoss : public ILoss
{
public:
	double Loss(Eigen::MatrixXd predict, Eigen::MatrixXd target) override;
	Eigen::MatrixXd LossPrime(Eigen::MatrixXd predict, Eigen::MatrixXd target) override;
};

