#include <iostream>
#include <Eigen/Dense>

#include "Layers/Linear.h"
#include "Loss/MSELoss.h"
#include "Layers/ActivationLayers/Sigmoid.h"

void main()
{
    Eigen::MatrixXd inputs(4, 1);
    inputs << 1.0, 2.0, 3.0, 2.5;

    Eigen::MatrixXd target(3, 1);
    target << 0, 1.0, 0.0;

    MSELoss loss;
    Sigmoid sigmoid;
    Linear linearRegression(4, 3);

    Eigen::MatrixXd predict; 
    predict = linearRegression.Forward(inputs);
    predict = sigmoid.Forward(predict);

    std::cout << "Initial prediction:\n" << predict << "\n\n";
    std::cout << "Initial loss:\n" << loss.Loss(predict, target) << "\n\n";

    for (int i = 0; i < 50; i++)
    {
        predict = linearRegression.Forward(inputs);
        predict = sigmoid.Forward(predict);
        predict = sigmoid.Forward(predict);

        Eigen::MatrixXd grad;
        grad = loss.LossPrime(predict, target);
        grad = sigmoid.Backward(grad, 0.001);
        linearRegression.Backward(grad, 0.001);
    }

    predict = linearRegression.Forward(inputs);
    predict = sigmoid.Forward(predict);

    std::cout << "Final prediction:\n" << predict << "\n\n";
    std::cout << "Final loss:\n" << loss.Loss(predict, target) << "\n\n";

}