#ifndef INEURAL_LAYERS_COMPRESSED_H_
#define INEURAL_LAYERS_COMPRESSED_H_

#include <iNeural/layers/Layer.h>
#include <iNeural/ActivationFuncs.h>
#include <iNeural/Regularization.h>

namespace iNeural
{

    class Compressed : public Layer
    {
        int I, J, M;
        bool bias;
        ActivationFunc act;
        double stdDev;
        Eigen::MatrixXd W;
        Eigen::MatrixXd Wd;
        Eigen::MatrixXd phi;
        Eigen::MatrixXd alpha;
        Eigen::MatrixXd alphad;
        Eigen::MatrixXd *x;
        Eigen::MatrixXd a;
        Eigen::MatrixXd y;
        Eigen::MatrixXd yd;
        Eigen::MatrixXd deltas;
        Eigen::MatrixXd e;
        Regularization regularization;

    public:
        Compressed(OutputInfo info, int J, int M, bool bias, ActivationFunc act, const std::string &compression, double stdDev, Regularization regularization);
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers);
        virtual void initializeParameters();
        virtual void updatedParameters();
        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
    };

} // namespace

#endif