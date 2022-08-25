#include "iNeural/Normalization.h"
#include "iNeural/util/AssertionMacros.h"

namespace iNeural
{
    Normalization::Normalization()
    {
    }

    Transformer &Normalization::fit(const Eigen::MatrixXd &X)
    {
        mean = X.colwise().mean();
        std.resize(mean.rows(), mean.cols());
        std.setZero();
        for (int n = 0; n < X.rows(); ++n)
        {
            for (int d = 0; d < X.cols(); ++d)
            {
                double tmp = X(n, d) - mean(d);
                std(0, d) += tmp * tmp;
            }
        }
        std /= X.rows();
        std.array() = std.array().sqrt();

        for (int d = 0; d < X.cols(); ++d)
            if (std(0, d) == 0.0)
                std(0, d) = 1.0;
        return *this;
    }

    Eigen::MatrixXd Normalization::transform(const Eigen::MatrixXd &X)
    {
        INEURAL_CHECK(mean.cols() > 0);
        INEURAL_CHECK_EQUALS(X.cols(), mean.cols());
        Eigen::MatrixXd normalized(X.rows(), X.cols());
        for (int n = 0; n < X.rows(); ++n)
            normalized.row(n).array() = (X.row(n).array() - mean.array()) *
                                        std.array().inverse();
        return normalized;
    }

    Eigen::VectorXd Normalization::getMean()
    {
        return mean.transpose();
    }

    Eigen::VectorXd Normalization::getStd()
    {
        return std.transpose();
    }

} // namespace iNeural