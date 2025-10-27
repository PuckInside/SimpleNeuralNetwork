# SimpleNeuralNetwork

SimpleNeuralNetwork - это небольшой проект для написания простых нейросетей на языке С++. В неё включено библиотека Eigen для работы с линейной алгеброй.

В проекте есть нейросеть вида Perceptron, которая реализует интерфейс INeuralNetwork. 
Для добавления нового функционала достаточно реализовать интерфейсы проекта. Например, ILayer:

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
