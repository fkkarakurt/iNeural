#ifndef INEURAL_COMPRESSOR_H_
#define INEURAL_COMPRESSOR_H_

#include <iNeural/Transformer.h>
#include <iNeural/CompressionMatrixFactory.h>
#include <Eigen/Core>

namespace iNeural
{
    class Compressor : public Transformer
    {
        Eigen::MatrixXd cm;

    public:
        Compressor(int inputDim, int outputDim, CompressionMatrixFactory::Transformation transformation);
        virtual Transformer &fit(const Eigen::MatrixXd &X);
        virtual Transformer fitPartial(const Eigen::MatrixXd &X);
        virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X);
        int getOutputs();
    };
} // namespace
#endif