#include "Perceptron.h"

Perceptron::Perceptron(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int hiddenLayerCount)
{
    this->layers.push_back(std::make_shared<Linear>(inputLayerSize, hiddenLayerSize));
    this->layers.push_back(std::make_shared<Sigmoid>());

    for (int i = 0; i < hiddenLayerCount; i++)
    {
        this->layers.push_back(std::make_shared<Linear>(hiddenLayerSize, hiddenLayerSize));
        this->layers.push_back(std::make_shared<Sigmoid>());
    }

    this->layers.push_back(std::make_shared<Linear>(hiddenLayerSize, outputLayerSize));
    this->layers.push_back(std::make_shared<Sigmoid>());
}

Eigen::MatrixXd Perceptron::Prediction(Eigen::MatrixXd inputs)
{
    Eigen::MatrixXd predict = inputs;
    for (auto layer : this->layers)
        predict = layer->Forward(predict);

    return predict;
}

void Perceptron::Fit(Eigen::MatrixXd inputs, Eigen::MatrixXd target, double learningRate, int epoch)
{
    Eigen::MatrixXd predict;
    Eigen::MatrixXd grad;

    for (int i = 0; i < epoch; i++)
    {
        predict = Prediction(inputs);
        grad = this->loss.LossPrime(predict, target);

        for (auto layer = this->layers.rbegin(); layer != this->layers.rend(); ++layer)
            grad = (*layer)->Backward(grad, learningRate);
    }
}
