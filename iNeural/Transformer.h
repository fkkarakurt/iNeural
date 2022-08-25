#ifndef INEURAL_TRANSFORMER_H_
#define INEURAL_TRANSFORMER_H_

#include <Eigen/Core>

namespace iNeural
{
    class Transformer
    {
    public:
        virtual ~Transformer() {}
        virtual Transformer &fit(const Eigen::MatrixXd &X) = 0;

        virtual Transformer &fitPartial(const Eigen::MatrixXd &X)
        {
            return fit(X);
        }

        virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X) = 0;
    };
} // namespace

#endif