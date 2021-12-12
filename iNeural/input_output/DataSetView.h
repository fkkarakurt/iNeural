#ifndef INEURAL_IO_DATA_SET_VIEW_H_
#define INEURAL_IO_DATA_SET_VIEW_H_

#include <Eigen/Core>
#include <vector>
#include <iNeural/input_output/DataSet.h>

namespace iNeural
{
    class Learner;

    class DataSetView : public DataSet
    {
    public:
        DataSetView(const DataSetView &dataset);
        DataSetView(DataSet &dataset) : dataset(&dataset) {}

        template <typename InputIt>
        DataSetView(DataSet &dataset, InputIt index_begin, InputIt index_end) : indices(index_begin, index_end), dataset(&dataset) {}

        virtual ~DataSetView() {}
        virtual int samples();
        virtual int inputs();
        virtual int outputs();
        virtual Eigen::VectorXd &getInstance(int i);
        virtual Eigen::VectorXd &getTarget(int i);
        virtual void finishIteration(Learner &learner);
        virtual DataSetView &shuffle();

    private:
        std::vector<int> indices;
        DataSet *dataset;
        friend void merge(DataSetView &merging, std::vector<DataSetView> &groups);
    };

    void split(std::vector<DataSetView> &groups, DataSet &dataset, int numberOfGroups, bool shuffling = true);
    void split(std::vector<DataSetView> &groups, DataSet &dataset, double ratio = 0.5, bool shuffling = true);
    void merge(DataSetView &merging, std::vector<DataSetView> &groups);
    DataSetView sample(DataSet &dataSet, double fraction, bool replacement);
} // namespace

#endif