#ifndef INEURAL_ERROR_FUNCS_H_
#define INEURAL_ERROR_FUNCS_H_

#include <Eigen/Core>

namespace iNeural
{
    template <typename Derived1, typename Derived2>
    double crossEntropy(const Eigen::MatrixBase<Derived1> &Y, const Eigen::MatrixBase<Derived2> &T)
    {
        return -(T.array() * ((Y.array() + 1e-10).log())).sum() / (double)Y.rows();
    }

    template <typename Derived>
    double meanSquaredError(const Eigen::MatrixBase<Derived> &YmT)
    {
        return YmT.array().square().sum() / (2.0 * (double)YmT.rows());
    }
}

#endif