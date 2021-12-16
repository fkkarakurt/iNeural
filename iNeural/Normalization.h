#ifndef INEURAL_NORMALIZATION_H_
#define INEURAL_NORMALIZATION_H_

#include <iNeural/Transformer.h>
#include <Eigen/Core>

namespace iNeural
{

    class Normalization : public Transformer
    {
        Eigen::MatrixXd mean;
        Eigen::MatrixXd std;

    public:
        Normalization();

        virtual Transformer &fit(const Eigen::MatrixXd &X);
        virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X);

        Eigen::VectorXd getMean();

        Eigen::VectorXd getStd();
    };

} // namespace

#endif