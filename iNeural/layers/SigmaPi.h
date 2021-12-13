#ifndef INEURAL_LAYERS_SIGMA_PI_H_
#define INEURAL_LAYERS_SIGMA_PI_H_

#include <Eigen/Core>
#include <vector>
#include <iNeural/layers/Layer.h>
#include <iNeural/ActivationFuncs.h>

namespace iNeural
{
    class SigmaPi : public Layer
    {
    protected:
        struct HigherOrderUnit
        {
            std::vector<int> position;
            size_t weight;
        };

        typedef std::vector<HigherOrderUnit> HigherOrderNeuron;

        OutputInfo info;
        bool bias;
        ActivationFunc act;
        double stdDev;

        Eigen::MatrixXd x;
        Eigen::MatrixXd a;
        Eigen::MatrixXd y;
        Eigen::MatrixXd yd;
        Eigen::MatrixXd deltas;
        Eigen::MatrixXd e;

        std::vector<double> w;
        std::vector<double> wd;
        std::vector<HigherOrderNeuron> nodes;

    public:
        SigmaPi(OutputInfo info, bool bias, ActivationFunc act, double stdDev);
        virtual OutputInfo initialize(std::vector<double *> &parameterPointers, std::vector<double *> &parameterDerivativePointers);

        struct Constraint
        {
            Constraint() {}
            virtual ~Constraint() {}
            virtual double operator()(int p1, int p2) const;
            virtual double operator()(int p1, int p2, int p3) const;
            virtual double operator()(int p1, int p2, int p3, int p4) const;
            virtual bool isDefault() const;
        };

        virtual SigmaPi &secondOrderNodes(int numbers);
        virtual SigmaPi &thirdOrderNodes(int numbers);
        virtual SigmaPi &fourthOrderNodes(int numbers);
        virtual SigmaPi &secondOrderNodes(int numbers, const Constraint &constrain);
        virtual SigmaPi &thirdOrderNodes(int numbers, const Constraint &constrain);
        virtual SigmaPi &fourthOrderNodes(int numbers, const Constraint &constrain);
        virtual size_t nodenumber() const { return nodes.size(); };
        virtual size_t parameter() const { return w.size(); };
        virtual void initializeParameters();
        virtual void updatedParameters();
        virtual void forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y, bool dropout = false, double *error = 0);
        virtual void backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout, bool backpropToPrevious);
        virtual Eigen::MatrixXd &getOutput();
        virtual Eigen::VectorXd getParameters();
    };
} // namespace

#endif