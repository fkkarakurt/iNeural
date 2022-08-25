#ifndef INEURAL_WEIGHTED_DATA_SET_H_
#define INEURAL_WEIGHTED_DATA_SET_H_

#include <iNeural/input_output/DataSet.h>
#include <vector>

namespace iNeural
{
    class WeightedDataSet : public DataSet
    {
        DataSet &dataSet;
        Eigen::VectorXd weights;
        bool deterministic;
        std::vector<int> originalIndices;

    public:
        WeightedDataSet(DataSet &dataSet, const Eigen::VectorXd &weights, bool deterministic);
        WeightedDataSet &updateWeights(const Eigen::VectorXd &weights);
        virtual int samples();
        virtual int inputs();
        virtual int outputs();
        virtual Eigen::VectorXd &getInstance(int n);
        virtual Eigen::VectorXd &getTarget(int n);
        virtual void finishIteration(Learner &learner) {}

    private:
        void resample();
    };
} // namespace

#endif
