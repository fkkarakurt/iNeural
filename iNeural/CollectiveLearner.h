#ifndef INEURAL_COLLECTIVE_H_
#define INEURAL_COLLECTIVE_H_

#include <iNeural/Learner.h>
#include <iNeural/optimization/Optimizer.h>
#include <iNeural/input_output/DataSet.h>
#include <Eigen/Core>

namespace iNeural
{
    class CollectiveLearner
    {
    public:
        virtual ~CollectiveLearner() {}
        virtual CollectiveLearner &addLearner(Learner &learner) = 0;
        virtual CollectiveLearner &setOptimizer(Optimizer &optimizer) = 0;
        virtual CollectiveLearner &train(DataSet &dataSet) = 0;
        virtual Eigen::MatrixXd operator()(Eigen::MatrixXd &X) = 0;
        virtual Eigen::MatrixXd operator()(Eigen::VectorXd &X) = 0;
    };
} // namespace

#endif