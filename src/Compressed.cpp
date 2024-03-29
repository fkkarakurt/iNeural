#include <iNeural/layers/Compressed.h>
#include <iNeural/CompressionMatrixFactory.h>
#include <iNeural/util/Random.h>

namespace iNeural
{

    Compressed::Compressed(OutputInfo info, int J, int M, bool bias,
                           ActivationFunc act, const std::string &compression,
                           double stdDev, Regularization regularization)
        : I(info.outputs()), J(J), M(M), bias(bias), act(act), stdDev(stdDev),
          W(J, I + bias), Wd(J, I + bias), phi(M, I + 1), alpha(J, M), alphad(J, M),
          x(0), a(1, J), y(1, J), yd(1, J), deltas(1, J), e(1, I + bias),
          regularization(regularization)
    {
        CompressionMatrixFactory::Transformation transformation =
            CompressionMatrixFactory::SPARSE_RANDOM;
        if (compression == std::string("gaussian"))
            transformation = CompressionMatrixFactory::GAUSSIAN;
        else if (compression == std::string("dct"))
            transformation = CompressionMatrixFactory::DCT;
        else if (compression == std::string("average"))
            transformation = CompressionMatrixFactory::AVERAGE;
        else if (compression == std::string("edge"))
            transformation = CompressionMatrixFactory::EDGE;
        CompressionMatrixFactory cmf(I + 1, M, transformation);
        cmf.createCompressionMatrix(phi);
    }

    OutputInfo Compressed::initialize(std::vector<double *> &parameterPointers,
                                      std::vector<double *> &parameterDerivativePointers)
    {
        parameterPointers.reserve(parameterPointers.size() + J * M);
        parameterDerivativePointers.reserve(parameterDerivativePointers.size() + J * M);
        for (int j = 0; j < J; j++)
        {
            for (int m = 0; m < M; m++)
            {
                parameterPointers.push_back(&alpha(j, m));
                parameterDerivativePointers.push_back(&alphad(j, m));
            }
        }

        initializeParameters();

        OutputInfo info;
        info.dimensions.push_back(J);
        return info;
    }

    void Compressed::initializeParameters()
    {
        RandomNumberGenerator rng;
        rng.fillNormalDistribution(alpha, stdDev);
        updatedParameters();
    }

    void Compressed::updatedParameters()
    {
        W = alpha * phi.block(0, 0, M, I + bias);
    }

    void Compressed::forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error)
    {
        const int N = x->rows();
        this->y.conservativeResize(N, Eigen::NoChange);
        this->x = x;
        a = *x * W.leftCols(I).transpose();
        if (bias)
            a.rowwise() += W.col(I).transpose();
        activationFunc(act, a, this->y);
        if (error && regularization.l1penal > 0.0)
            *error += regularization.l1penal * alpha.array().abs().sum();
        if (error && regularization.l2penal > 0.0)
            *error += regularization.l2penal * alpha.array().square().sum() / 2.0;
        y = &(this->y);
    }

    void Compressed::backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious)
    {
        const int N = a.rows();
        yd.conservativeResize(N, Eigen::NoChange);
        activationFuncDerivative(act, y, yd);
        deltas = yd.cwiseProduct(*ein);
        Wd.leftCols(I) = deltas.transpose() * *x;
        alphad = Wd.leftCols(I) * phi.block(0, 0, M, I).transpose();
        if (bias)
        {
            Wd.col(I) = deltas.colwise().sum().transpose();
            alphad += Wd.col(I) * phi.col(I).transpose();
        }
        if (regularization.l1penal > 0.0)
            alphad.array() += regularization.l1penal * alpha.array() / alpha.array().abs();
        if (regularization.l2penal > 0.0)
            alphad += regularization.l2penal * alpha;
        if (backpropToPrevious)
            e = deltas * W.leftCols(I);
        eout = &e;
    }

    Eigen::MatrixXd &Compressed::getOutput()
    {
        return y;
    }

    Eigen::VectorXd Compressed::getParameters()
    {
        Eigen::VectorXd p(M * I);
        int idx = 0;
        for (int m = 0; m < M; m++)
            for (int i = 0; i < I; i++)
                p(idx++) = phi(m, i);
        return p;
    }

} // namespace