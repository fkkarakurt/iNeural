#ifndef INEURAL_INPUT_OUTPUT_DATA_SET_H_
#define INEURAL_INPUT_OUTPUT_DATA_SET_H_

#include <Eigen/Core>

namespace iNeural
{
    class Learner;
    class DataSet
    {
    public:
        virtual ~DataSet() {}
        virtual int samples() = 0;
        virtual int inputs() = 0;
        virtual int outputs() = 0;
        virtual Eigen::VectorXd &getInstance(int n) = 0;
        virtual Eigen::VectorXd &getTarget(int n) = 0;
        virtual void finishIteraction(Learner &learner) = 0;
    };
} // namespace

#endif