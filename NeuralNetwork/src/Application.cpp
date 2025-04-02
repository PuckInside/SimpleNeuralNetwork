#include <iostream>
#include <Eigen/Dense>

#include "Layers/Linear.h"
#include "Loss/MSELoss.h"

void main()
{
    Eigen::MatrixXd inputs(4, 1);
    inputs << 1.0, 2.0, 3.0, 2.5;

    Eigen::MatrixXd target(3, 1);
    target << 4.0, 2.0, 1.0;

    MSELoss loss;
    Linear linearRegression(4, 3);
    Eigen::MatrixXd predict = linearRegression.Forward(inputs);
    std::cout << "Initial prediction:\n" << predict << "\n\n";

    for (int i = 0; i < 28; i++)
    {
        predict = linearRegression.Forward(inputs);
        Eigen::MatrixXd grad = loss.LossPrime(predict, target);
        
        linearRegression.Backward(grad, 0.01);
    }

    predict = linearRegression.Forward(inputs);
    std::cout << "Final prediction:\n" << predict << "\n";
}