#include "MSELoss.h"

double MSELoss::Loss(Eigen::MatrixXd predict, Eigen::MatrixXd target)
{
    return (predict - target).array().square().mean();
}

Eigen::MatrixXd MSELoss::LossPrime(Eigen::MatrixXd predict, Eigen::MatrixXd target)
{
    Eigen::MatrixXd msePrime = 2 * (predict - target) / predict.cols();
    return msePrime;
}
