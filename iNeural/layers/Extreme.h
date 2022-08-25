#ifndef INEURAL_LAYERS_EXTREME_H_
#define INEURAL_LAYERS_EXTREME_H_

#include <iNeural/layers/Layer.h>
#include <iNeural/ActivationFuncs.h>

namespace iNeural
{

    class Extreme : public Layer
    {
        int I, J;
        bool bias;
        ActivationFunc act;
        double stdDev;
        Eigen::MatrixXd W;
        Eigen::MatrixXd *x;
        Eigen::MatrixXd a;
        Eigen::MatrixXd y;
        Eigen::MatrixXd yd;
        Eigen::MatrixXd deltas;
        Eigen::MatrixXd e;

    public:
        Extreme(OutputInfo info, int J, bool bias, ActivationFunc act, double stdDev);
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