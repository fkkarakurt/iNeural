#ifndef INEURAL_PCA_H_
#define INEURAL_PCA_H_

#include <iNeural/Transformer.h>
#include <Eigen/Core>

namespace iNeural
{
    class PCA : public Transformer
    {
        int components;
        bool whiten;
        Eigen::VectorXd mean;
        Eigen::MatrixXd W;
        Eigen::VectorXd evr;

    public:
        PCA(int components, bool whiten = true);

        virtual Transformer &fit(const Eigen::MatrixXd &X);
        virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X);

        Eigen::VectorXd explainedVarianceRatio();
    };
} // namespace

#endif