#ifndef INEURAL_LAYERS_LAYER_H_
#define INEURAL_LAYERS_LAYER_H_

#include <Eigen/Core>
#include <vector>

namespace iNeural
{
    class OutputInfo
    {
    public:
        std::vector<int> dimensions;
        int outputs();
    };

    class Layer
    {
    public:
        virtual ~Layer() {}
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers) = 0;
        virtual void initializeParameters() = 0;
        virtual void updatedParameters() = 0;
        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error = 0) = 0;
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious) = 0;
        virtual Eigen::MatrixXd &getOutput() = 0;
        virtual Eigen::VectorXd getParameters() = 0;
    };

} // namespace

#endif