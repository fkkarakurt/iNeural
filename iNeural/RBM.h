#ifndef INEURAL_RBM_H_
#define INEURAL_RBM_H_

#include <iNeural/Learner.h>
#include <iNeural/Regularization.h>
#include <iNeural/layers/Layer.h>
#include <iNeural/optimization/Optimizable.h>
#include <iNeural/optimization/StoppingCriteria.h>
#include <Eigen/Core>
#include <vector>

namespace iNeural
{

    class RandomNumberGenerator;

    class RBM : public Learner, public Layer
    {
        RandomNumberGenerator *rng;
        int D, H;
        int cdN;
        double stdDev;
        Eigen::MatrixXd W, posGradW, negGradW, Wd;
        Eigen::VectorXd bv, posGradBv, negGradBv, bh, posGradBh, negGradBh, bhd;
        Eigen::MatrixXd pv, v, ph, h, phd;
        Eigen::MatrixXd deltas, e;
        int K;
        Eigen::VectorXd params, grad;
        bool backprop;
        Regularization regularization;

    public:
        RBM(int D, int H, int cdN = 1, double stdDev = 0.01, bool backprop = true, Regularization regularization = Regularization());
        virtual ~RBM();

        virtual Eigen::VectorXd operator()(const Eigen::VectorXd &x);
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &X);
        virtual bool providesInitialization();
        virtual void initialize();
        virtual unsigned int examples();
        virtual unsigned int dimension();
        virtual void setParameters(const Eigen::VectorXd &parameters);
        virtual const Eigen::VectorXd &currentParameters();
        virtual double error();
        virtual double error(unsigned int n);
        virtual bool providesGradient();
        virtual Eigen::VectorXd gradient();
        virtual Eigen::VectorXd gradient(unsigned int n);
        virtual void errorGradient(std::vector<int>::const_iterator startN, std::vector<int>::const_iterator endN, double &value, Eigen::VectorXd &grad);

        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers);
        virtual void initializeParameters() {}
        virtual void updatedParameters() {}

        int visibleUnits();

        int hiddenUnits();

        const Eigen::MatrixXd &getWeights();

        const Eigen::MatrixXd &getVisibleProbs();

        const Eigen::MatrixXd &getVisibleSample();

        Eigen::MatrixXd reconstructProb(int n, int steps);

        void sampleHgivenV();

        void sampleVgivenH();

    private:
        void reality();
        void daydream();
        void fillGradient();
    };

} // namespace OpenANN

#endif