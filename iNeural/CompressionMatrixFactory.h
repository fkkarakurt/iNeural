#ifndef INEURAL_COMPRESSION_MATRIX_FACTORY_H_
#define INEURAL_COMPRESSION_MATRIX_FACTORY_H_

#include <Eigen/Core>
#include <vector>

namespace iNeural
{
    class CompressionMatrixFactory
    {
    public:
        bool compress;
        enum Transformation
        {
            DCT,
            GAUSSIAN,
            SPARSE_RANDOM,
            AVERAGE,
            EDGE
        } transformation;
        int inputDim;
        int paramDim;

        CompressionMatrixFactory();
        CompressionMatrixFactory(int inputDim, int paramDim, Transformation transformation = DCT);
        void createCompressionMatrix(Eigen::MatrixXd &cm);

    private:
        void fillCompressionMatrix(Eigen::MatrixXd &cm);
    };
} // namespace
#endif