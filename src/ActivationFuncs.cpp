#include <iNeural/ActivationFuncs.h>
#include <limits>
#include <cmath>

namespace iNeural
{
    void activationFunc(ActivationFunc activate, const Eigen::MatrixXd &a, Eigen::MatrixXd &z)
    {
        switch (activate)
        {
        case LOGISTIC:
            logistic(a, z);
            break;
        case TANH:
            tanh(a, z);
            break;
        case TANH_SCALED:
            scaledtanh(a, z);
            break;
        case RECTIFIER:
            rectifier(a, z);
            break;
        case LINEAR:
        default:
            linear(a, z);
            break;
        }
    } // void activationFunc

    void activationFuncDerivative(ActivationFunc activate, const Eigen::MatrixXd &z, Eigen::MatrixXd &gd)
    {
        switch (activate)
        {
        case LOGISTIC:
            logisticDerivative(z, gd);
            break;
        case TANH:
            tanhDerivative(z, gd);
            break;
        case TANH_SCALED:
            scaledtanhDerivative(z, gd);
            break;
        case RECTIFIER:
            rectifierDerivative(z, gd);
            break;
        case LINEAR:
        default:
            linearDerivative(gd);
            break;
        }
    } // void activationFuncDerivative

    void softmax(Eigen::MatrixXd &y)
    {
        const int N = y.rows();
        const double max = y.maxCoeff();
        for (int i = 0; i < N; i++)
        {
            y.row(i) = (y.row(i).array() - max).exp();
            y.row(i) /= y.row(i).sum();
        }
    } // void softmax

    void logistic(const Eigen::MatrixXd &a, Eigen::MatrixXd &z)
    {
        double const *aPtr = a.data();
        double const *aEnd = aPtr + a.rows() * a.cols();
        for (double *zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
        {
            if (*aPtr < -45.0)
                *zPtr = 0.0;
            else if (*aPtr > 45.0)

                *zPtr = 1.0;
            else
                *zPtr = 1.0 / (1.0 + std::exp(-*aPtr));
        }
    } // void logistic

    void logisticDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd)
    {
        double const *zPtr = z.data();
        double const *zEnd = zPtr + z.rows() * z.cols();
        for (double *gdPtr = gd.data(); zPtr < zEnd; zPtr++, gdPtr++)
            *gdPtr = *zPtr * (1.0 - *zPtr);
    } // void logisticDerivative

    void tanh(const Eigen::MatrixXd &a, Eigen::MatrixXd &z)
    {
        double const *aPtr = a.data();
        double const *aEnd = aPtr + a.rows() * a.cols();
        for (double *zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
            *zPtr = std::tanh(*aPtr);
    } // void tanh

    void tanhDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd)
    {
        double const *zPtr = z.data();
        double const *zEnd = zPtr + z.rows() * z.cols();
        for (double *gdPtr = gd.data(); zPtr < zEnd; zPtr++, gdPtr++)
            *gdPtr = 1.0 - *zPtr * *zPtr;
    } // void tanhDerivative

    void scaledtanh(const Eigen::MatrixXd &a, Eigen::MatrixXd &z)
    {
        double const *aPtr = a.data();
        double const *aEnd = aPtr + a.rows() * a.cols();
        for (double *zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
            *zPtr = 1.7159 * std::tanh(0.66666667 * *aPtr);
    } // void scaledtanh

    void scaledtanhDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd)
    {
        double const *zPtr = z.data();
        double const *zEnd = zPtr + z.rows() * z.cols();
        for (double *gdPtr = gd.data(); zPtr < zEnd; zPtr++, gdPtr++)
            *gdPtr = 0.66666667 / 1.7159 * (1.7159 + *zPtr) * (1.7159 - *zPtr);
    } // void scaledtanhDerivative

    void rectifier(const Eigen::MatrixXd &a, Eigen::MatrixXd &z)
    {
        double const *aPtr = a.data();
        double const *aEnd = aPtr + a.rows() * a.cols();
        for (double *zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
            *zPtr = std::max<double>(0.0, *aPtr);
    } // void rectifier

    void rectifierDerivative(const Eigen::MatrixXd &z, Eigen::MatrixXd &gd)
    {
        double const *zPtr = z.data();
        double const *zEnd = zPtr + z.rows() * z.cols();
        for (double *gdPtr = gd.data(); zPtr < zEnd; zPtr++, gdPtr++)
            *gdPtr = (double)(*zPtr > 0.0) * 1.0;
    } // void rectifierDerivative

    void linear(const Eigen::MatrixXd &a, Eigen::MatrixXd &z)
    {
        z = a;
    } // void linear

    void linearDerivative(Eigen::MatrixXd &gd)
    {
        gd.fill(1.0);
    } // void linearDerivative

} // namespace iNeural