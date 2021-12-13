#ifndef INEURAL_LAYERS_MAX_POOLING_H_
#define INEURAL_LAYERS_MAX_POOLING_H_

#include <iNeural/layers/Layer.h>
#include <iNeural/ActivationFuncs.h>

namespace iNeural
{
    class MaxPooling : public Layer
    {
        int I, fm, inRows, inCols, kernelRows, kernelCols;
        Eigen::MatrixXd *x;
        Eigen::MatrixXd y;
        Eigen::MatrixXd e;
        int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;

    public:
        MaxPooling(OutputInfo info, int kernelRows, int kernelCols);
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers);
        virtual void initializeParameters();
        virtual void updatedParameters() {}
        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
    };
} // namespace

#endif