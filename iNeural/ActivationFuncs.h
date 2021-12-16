#ifndef INEURAL_ACTIVATION_FUNCS_H_
#define INEURAL_ACTIVATION_FUNCS_H_

#include <Eigen/Core>

namespace iNeural
{
    enum ActivationFunc
    {
        LOGISTIC = 0,

        TANH = 1,

        TANH_SCALED = 2,

        RECTIFIER = 3,

        LINEAR = 4,

        SOFTMAX = 4
    };

    void activationFunc(ActivationFunc activate, const Eigen::MatrixXd &a, Eigen::MatrixXd &z);
    void activationFuncDerivative(ActivationFunc activate, const Eigen::MatrixXd &z, Eigen::MatrixXd &gd);
    void softmax(Eigen::MatrixXd &y);
    void logistic(const Eigen::MatrixXd &a, Eigen::MatrixXd &z);
    void logisticDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd);
    void tanh(const Eigen::MatrixXd &a, Eigen::MatrixXd &z);
    void tanhDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd);
    void scaledtanh(const Eigen::MatrixXd &a, Eigen::MatrixXd &z);
    void scaledtanhDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd);
    void rectifier(const Eigen::MatrixXd &a, Eigen::MatrixXd &z);
    void rectifierDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd);
    void linear(const Eigen::MatrixXd &a, Eigen::MatrixXd &z);
    void linearDerivative(Eigen::MatrixXd &gd);
}

#endif