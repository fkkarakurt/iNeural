#ifndef INEURAL_INPUT_OUTPUT_DATA_STREAM_H_
#define INEURAL_INPUT_OUTPUT_DATA_STREAM_H_

#include <Eigen/Core>

namespace iNeural
{
    class DirectStorageDataSet;
    class Optimizer;
    class Learner;

    class DataStream
    {
        int cacheSize, collected;
        Eigen::MatrixXd X, T;
        DirectStorageDataSet *cache;
        Optimizer *opt;   // DON'T DELETE
        Learner *learner; // DON'T DELETE
    public:
        DataStream(int cacheSize);
        ~DataStream();

        DataStream &setLearner(Learner &learner);
        DataStream &setOptimizer(Optimizer &opt);

        void addSample(Eigen::VectorXd *x, Eigen::VectorXd *t = 0);

    private:
        void initialize(int inputs, int outputs);
    };
} // namespace

#endif