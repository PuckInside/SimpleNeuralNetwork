#include <iostream>
#include <Eigen/Dense>

#include <vector>
#include <memory>

#include "Models/Perceptron.h"
#include "Loss/MSELoss.h"

void main()
{
    Eigen::MatrixXd inputs(4, 1);
    inputs << 1.0, 2.0, 3.0, 2.5;

    Eigen::MatrixXd target(3, 1);
    target << 0, 1.0, 0.0;

    Perceptron model(4, 120, 3, 2);
    MSELoss loss;

    Eigen::MatrixXd predict;
    predict = model.Prediction(inputs);

    std::cout << "Initial prediction:\n" << predict << "\n\n";
    std::cout << "Initial loss:\n" << loss.Loss(predict, target) << "\n\n";

    model.Fit(inputs, target, 0.01, 10000);
    predict = model.Prediction(inputs);

    std::cout << "Final prediction:\n" << predict << "\n\n";
    std::cout << "Final loss:\n" << loss.Loss(predict, target) << "\n\n";
}