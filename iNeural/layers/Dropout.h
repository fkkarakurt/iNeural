#ifndef INEURAL_LAYERS_DROPOUT_H_
#define INEURAL_LAYERS_DROPOUT_H_

#include <iNeural/layers/Layer.h>

namespace iNeural
{

    class Dropout : public Layer
    {
        OutputInfo info;
        int I;
        double dropoutProbability;
        Eigen::MatrixXd dropoutMask;
        Eigen::MatrixXd y;
        Eigen::MatrixXd e;

    public:
        Dropout(OutputInfo info, double dropoutProbability);
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers,
                                      std::vector<double *> &parameterDerivativePointers);
        virtual void initializeParameters() {}
        virtual void updatedParameters() {}
        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y,
                                      bool dropout, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout,
                                   bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
    };

}

#endif