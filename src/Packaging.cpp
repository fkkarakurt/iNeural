#include <iNeural/Packaging.h>
#include <iNeural/util/AssertionMacros.h>
#include <iNeural/input_output/DataSetView.h>

namespace iNeural
{

    Packaging::Packaging(double packageSize)
        : packageSize(packageSize), F(0)
    {
    }

    CollectiveLearner &Packaging::addLearner(Learner &learner)
    {
        models.push_back(&learner);
    }

    CollectiveLearner &Packaging::setOptimizer(Optimizer &optimizer)
    {
        this->optimizer = &optimizer;
    }

    CollectiveLearner &Packaging::train(DataSet &dataSet)
    {
        const int N = dataSet.samples();
        F = dataSet.outputs();
        for (std::list<Learner *>::iterator m = models.begin(); m != models.end(); m++)
        {
            DataSetView samples = sample(dataSet, bagSize, true);
            (*m)->trainingSet(samples);
            optimizer->setOptimizable(**m);
            optimizer->optimize();
            (*m)->removeTrainingSet();
        }
    }

    Eigen::MatrixXd Packaging::operator()(Eigen::MatrixXd &X)
    {
        const int N = X.rows();
        Eigen::MatrixXd Y(N, F);
        Y.fill(0.0);

        for (std::list<Learner *>::iterator m = models.begin(); m != models.end(); m++)
            Y += (**m)(X);

        return Y / models.size();
    }

    Eigen::VectorXd Packaging::operator()(Eigen::VectorXd &x)
    {
        Eigen::MatrixXd X = X.transpose();
        return (*this)(X).transpose();
    }

}