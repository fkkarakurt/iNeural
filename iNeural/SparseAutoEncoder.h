#ifndef INEURAL_SPARSE_AUTO_ENCODER_H_
#define INEURAL_SPARSE_AUTO_ENCODER_H_

#include <iNeural/Learner.h>
#include <iNeural/ActivationFuncs.h>
#include <iNeural/layers/Layer.h>

namespace iNeural
{

    class SparseAutoEncoder : public Learner, public Layer
    {
        int D, H;
        double beta, rho, lambda;
        ActivationFunc act;
        Eigen::MatrixXd X;
        Eigen::MatrixXd W1, W2, W1d, W2d;
        Eigen::VectorXd b1, b2, b1d, b2d;
        Eigen::MatrixXd A1, Z1, G1D, A2, Z2, G2D;
        Eigen::VectorXd parameters, grad;
        Eigen::MatrixXd dEdZ2, dEdZ1;
        Eigen::VectorXd meanActivation;

    public:
        SparseAutoEncoder(int D, int H, double beta, double rho, double lambda, ActivationFunc act);

        virtual Eigen::VectorXd operator()(const Eigen::VectorXd &x);
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &X);
        virtual bool providesInitialization();
        virtual void initialize();
        virtual unsigned int dimension();
        virtual void setParameters(const Eigen::VectorXd &parameters);
        virtual const Eigen::VectorXd &currentParameters();
        virtual double error();
        virtual bool providesGradient();
        virtual Eigen::VectorXd gradient();
        virtual void errorGradient(double &value, Eigen::VectorXd &grad);
        virtual Learner &trainingSet(DataSet &trainingSet);

        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers);
        virtual void initializeParameters();
        virtual void updatedParameters() {}

        Eigen::MatrixXd getInputWeights();
        Eigen::MatrixXd getOutputWeights();
        Eigen::VectorXd reconstruct(const Eigen::VectorXd &x);
    };

} // namespace

#endif