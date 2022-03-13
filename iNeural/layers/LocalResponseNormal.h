#ifndef INEURAL_LAYERS_LOCAL_RESPONSE_NORMAL_H_
#define INEURAL_LAYERS_LOCAL_RESPONSE_NORMAL_H_

#include <iNeural/layers/Layer.h>

namespace iNeural
{

    class LocalResponseNormal : public Layer
    {
        int I, fm, rows, cols;
        int fmSize;
        double k;
        int n;
        double alpha;
        double beta;
        Eigen::MatrixXd *x;
        Eigen::MatrixXd denoms;
        Eigen::MatrixXd y;
        Eigen::MatrixXd etmp;
        Eigen::MatrixXd e;

    public:
        LocalResponseNormal(OutputInfo info, double k, int n, double alpha,
                            double beta);
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