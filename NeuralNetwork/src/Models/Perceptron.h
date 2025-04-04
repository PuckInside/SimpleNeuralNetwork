#pragma once
#include <memory>
#include <vector>

#include "../Interfaces/INeuralNetwork.h"
#include "../Interfaces/ILayer.h"
#include "../Interfaces/ILoss.h"

#include "../Layers/Linear.h"
#include "../Layers/ActivationLayers/Sigmoid.h"
#include "../Loss/MSELoss.h"

class Perceptron : public INeuralNetwork
{
private:
	std::shared_ptr<ILoss> loss;
	std::vector<std::shared_ptr<ILayer>> layers;

public:
	Perceptron(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int hiddenLayerCount);

	Eigen::MatrixXd Prediction(Eigen::MatrixXd inputs) override;
	void Fit(Eigen::MatrixXd inputs, Eigen::MatrixXd target, double learningRate, int epoch) override;
};

