#ifndef INEURAL_LAYERS_ALPHA_BETA_FILTER_H_
#define INEURAL_LAYERS_ALPHA_BETA_FILTER_H_

#include <iNeural/layers/Layer.h>
#include <iNeural/ActivationFuncs.h>

namespace iNeural
{

    class AlphaBetaFilter : public Layer
    {
        int I, J;
        double deltaT;
        double stdDev;
        Eigen::VectorXd gamma;
        Eigen::VectorXd gammad;
        Eigen::VectorXd alpha;
        Eigen::VectorXd beta;
        bool first;
        Eigen::MatrixXd *x;
        Eigen::MatrixXd y;

    public:
        AlphaBetaFilter(OutputInfo info, double deltaT, double stdDev);
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers,
                                      std::vector<double *> &parameterDerivativePointers);
        virtual void initializeParameters();
        virtual void updatedParameters();
        virtual void reset();
        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y,
                                      bool dropout, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout,
                                   bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
    };

}

#endif