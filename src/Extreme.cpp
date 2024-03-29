#include <iNeural/layers/Extreme.h>
#include <iNeural/util/Random.h>

namespace iNeural
{

    Extreme::Extreme(OutputInfo info, int J, bool bias, ActivationFunction act, double stdDev)
        : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
          W(J, I + bias), x(0), a(1, J), y(1, J), yd(1, J), deltas(1, J), e(1, I)
    {
    }

    OutputInfo Extreme::initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers)
    {
        initializeParameters();

        OutputInfo info;
        info.dimensions.push_back(J);
        return info;
    }

    void Extreme::initializeParameters()
    {
        RandomNumberGenerator rng;
        rng.fillNormalDistribution(W, stdDev);
    }

    void Extreme::forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error)
    {
        this->x = x;
        a = *x * W.leftCols(I).transpose();
        if (bias)
            a.rowwise() += W.col(I).transpose();
        this->y.conservativeResize(a.rows(), Eigen::NoChange);
        activationFunction(act, a, this->y);
        y = &(this->y);
    }

    void Extreme::backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious)
    {
        this->yd.conservativeResize(a.rows(), Eigen::NoChange);
        activationFunctionDerivative(act, y, yd);
        deltas = yd.cwiseProduct(*ein);
        if (backpropToPrevious)
            e = deltas * W.leftCols(I);
        eout = &e;
    }

    Eigen::MatrixXd &Extreme::getOutput()
    {
        return y;
    }

    Eigen::VectorXd Extreme::getParameters()
    {
        return Eigen::VectorXd();
    }

}