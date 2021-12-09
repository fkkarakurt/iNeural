#ifndef INEURAL_ASSESSMENT_H_
#define INEURAL_ASSESSMENT_H_

#include <Eigen/Core>

namespace iNeural
{
    class Optimizer;
    class Learner;
    class DataSet;

    double sse(Learner &learner, DataSet &dataSet);
    double mse(Learner &learner, DataSet &dataSet);
    double rmse(Learner &learner, DataSet &dataSet);
    double ce(Learner &learner, DataSet &dataSet);
    double accuracy(Learner &learner, DataSet &dataSet);
    double weightedAccuracy(Learner &learner, DataSet &dataSet, Eigen::VectorXd weights);

    Eigen::MatrixXi confusionMatrix(Learner &learner, DataSet &dataSet);

    int classificationHits(Learner &learner, DataSet &dataSet);

    double crossValidation(int folds, Learner &learner, DataSet &dataSet, Optimizer &opt);
    int oneOfCDecoding(const Eigen::VectorXd &target);

} // namespace

#endif