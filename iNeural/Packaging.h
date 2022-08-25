#ifndef INEURAL_PACKAGING_H_
#define INEURAL_PACKAGING_H

#include <iNeural/CollectiveLearner.h>
#include <list>

namespace iNeural
{
    class Packaging : public CollectiveLearner
    {
        std::list<Learner *> models;
        Optimizer *optimizer;
        double packageSize;
        int F;

    public:
        Packaging(double packageSize);
        virtual CollectiveLearner &addLearner(Learner &learner);
        virtual CollectiveLearner &setOptimizer(Optimizer &optimizer);
        virtual CollectiveLearner &train(DataSet &dataSet);
        virtual Eigen::MatrixXd operator()(Eigen::MatrixXd &X);
        virtual Eigen::VectorXd operator()(Eigen::VectorXd &x);
    };
} // namespace

#endif