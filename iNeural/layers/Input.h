#ifndef INEURAL_LAYERS_INPUT_H_
#define INEURAL_LAYERS_INPUT_H_

#include <iNeural/layers/Layer.h>
#include <iNeural/ActivationFuncs.h>

namespace iNeural
{
    class Input : public Layer
    {
        int J, dim1, dim2, dim3;
        Eigen::MatrixXd *x;

    public:
        Input(int dim1, int dim2, int dim3);
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