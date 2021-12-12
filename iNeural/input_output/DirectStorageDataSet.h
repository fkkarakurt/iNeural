#ifndef INEURAL_INPUT_OUTPUT_DIRECT_STORAGE_DATA_SET_H_
#define INEURAL_INPUT_OUTPUT_DIRECT_STORAGE_DATA_SET_H_
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunushed-parameter"
#endif

#include <iNeural/input_output/DataSet.h>
#include <iNeural/input_output/Logger.h>

namespace iNeural
{
    class Evaluator;

    class DirectStorageDataSet : public DataSet
    {
    protected:
        Eigen::MatrixXd *in;
        Eigen::MatrixXd *out;
        const int N;
        const int D;
        const int F;
        Eigen::VectorXd temporaryInput;
        Eigen::VectorXd temporaryOutput;
        Evaluator *evaluator; // DON'T DELETE

    public:
        DirectStorageDataSet(Eigen::MatrixXd *in, Eigen::MatrixXd *out = 0, Evaluator *evaluator = 0);
        virtual int samples() { return N; }
        virtual int inputs() { return D; }
        virtual int outputs() { return F; }
        virtual Eigen::VectorXd &getInstance(int i);
        virtual Eigen::VectorXd &getTarget(int i);
        virtual void finishIteration(Learner &learner);
    };
} // namespace

#endif