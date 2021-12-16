#ifndef INEURAL_ZCA_WHITENING_H_
#define INEURAL_ZCA_WHITENING_H_

#include <iNeural/Transformer.h>
#include <Eigen/Core>

namespace iNeural
{

    class ZeroCAWhitening : public Transformer
    {
        Eigen::VectorXd mean;
        Eigen::MatrixXd W;

    public:
        virtual Transformer &fit(const Eigen::MatrixXd &X);
        virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X);
    };

} // namespace
#endif