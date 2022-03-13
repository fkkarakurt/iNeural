#ifndef INEURAL_OPTIMIZATION_MBSGD_H_
#define INEURAL_OPTIMIZATION_MBSGD_H_

#include <iNeural/optimization/Optimizer.h>
#include <iNeural/optimization/StoppingCriteria.h>
#include <iNeural/util/Random.h>
#include <Eigen/Core>
#include <vector>
#include <list>

namespace iNeural
{
    class MiniBSGD : public Optimizer
    {
        StoppingCriteria stop;
        Optimizable *opt; // DON'T DELETE
        bool nesterov;
        double alpha;
        double alphaDecay;
        double minAlpha;
        double eta;
        double etaGain;
        double maxEta;
        int batchSize;
        double minGain;
        double maxGain;
        bool useGain;
        int iteration;
        RandomNumberGenerator rng;
        int P, N, batches;
        Eigen::VectorXd gradient, gains, parameters, momentum, currentGradient;
        double accumulatedError;
        std::vector<int> randomIndices;

    public:
        MiniBSGD(double learningRate = 0.01, double momentum = 0.5, int batchSize = 10,
                 bool nesterov = false, double learningRateDecay = 1.0,
                 double minimalLearningRate = 0.0, double momentumGain = 0.0,
                 double maximalMomentum = 1.0, double minGain = 1.0,
                 double maxGain = 1.0);
        ~MiniBSGD();
        virtual void setOptimizable(Optimizable &opt);
        virtual void setStopCriteria(const StoppingCriteria &stop);
        virtual void optimize();
        virtual bool step();
        virtual Eigen::VectorXd result();
        virtual std::string name();

    private:
        void initialize();
    };
}

#endif