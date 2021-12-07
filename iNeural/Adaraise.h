#ifndef INEURAL_ADARAISE_H_
#define INEURAL_ADARAISE_H_

#include <iNeural/CollectiveLearner.h>
#include <list>

namespace iNeural
{
    class AdaRaise : public CollectiveLearner
    {
        std::list<Learner *> models;
        Optimizer *optimizer;
        Eigen::VectorXd modelWeights;
        int F;

    public:
        AdaRaise();
        Eigen::VectorXd getWeights();
        virtual CollectiveLearner &addLearner(Learner &learner);
        virtual CollectiveLearner &setOptimizer(Optimizer &optimizer);
        virtual CollectiveLearner &train(DataSet &dataSet);
        virtual Eigen::MatrixXd operator()(Eigen::MatrixXd &X);
        virtual Eigen::VectorXd operator()(Eigen::VectorXd &x);
    };
} // namespace

#endif