#ifndef INEURAL_LEARNER_H_
#define INEURAL_LEARNER_H_

#include <iNeural/input_output/DataSet.h>
#include <iNeural/optimization/Optimizable.h>

namespace iNeural
{
    class Learner : public Optimizable
    {
    protected:
        DataSet *trainSet;
        DataSet *validSet;
        bool deleteTrainSet, deleteValidSet;
        int N;

    public:
        Learner();
        virtual ~Learner();
        virtual Eigen::VectorXd operator()(const Eigen::VectorXd &x) = 0;
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &X) = 0;
        virtual Learner &trainingSet(Eigen::MatrixXd &input, Eigen::MatrixXd &output);
        virtual Learner &trainingSet(DataSet &trainingSet);
        virtual Learner &removeTrainingSet();
        virtual Learner &validationSet(Eigen::MatrixXd &input, Eigen::MatrixXd &output);
        virtual Learner &validationSet(DataSet &validationSet);
        virtual Learner &removeValidationSet();
    };
} // namespace

#endif