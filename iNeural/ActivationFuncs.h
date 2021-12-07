#ifndef INEURAL_ACTIVATION_FUNCS_H_
#define INEURAL_ACTIVATION_FUNCS_H_

#include <Eigen/Core>

namespace iNeural
{
    enum ActivationFunc
    {
        // Logistic Sigmoid Activation Func.
        // Range => [0,1]
        LOGISTIC = 0,

        // Tanh Sigmoid Func.
        // Range => [0,1]
        TANH = 1,

        // Scaled Tanh Sigmoid Func.
        // Range => [-1.7159, 1.7159]
        TANH_SCALED = 2,

        // non-saturating Rectified Linear Unit (ReLU)
        // Range => [0, \f$ \infty \f$]
        RECTIFIER = 3,

        // Identity Func.
        // Range => [\f$ -\infty \f$, \f$ \infty \f$]
        LINEAR = 4,

        // SoftMax Activation Func.
        // Range => [0,1]
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