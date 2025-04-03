# SimpleNeuralNetwork

SimpleNeuralNetwork - это небольшой инструмент для написания нейросети на языке С++. В проект включена библиотека Eigen, чтобы работать с линейной алгеброй.

В проекте есть готовая нейросеть, которая называется Perceptron, она реализует интерфейс INeuralNetwork. Она достаточно грубо сделана, например, в ней нельзя сделать скрытые слои с произвольным размером. Она предназдначена для примера.

Если вам необходимо добавить новый функционал, то реализуйте имеющиеся интерфейсы. Например, ILayer для слоя активаций:

```cpp
#include "ILayer.h"
#include <Eigen/Dense>

class Softmax : public ILayer
{
public:
    Eigen::MatrixXd Forward(Eigen::MatrixXd inputs) override
    {
        Eigen::MatrixXd expValues = inputs.array().exp();
        Eigen::MatrixXd sums = expValues.rowwise().sum();
        return expValues.array().colwise() / sums.array();
    }

    Eigen::MatrixXd Backward(Eigen::MatrixXd outputGradient, double learningRate) override
    {
        return outputGradient;
    }
};
```
