#ifndef INEURAL_INPUT_OUTPUT_LIB_SVM_H_
#define INEURAL_INPUT_OUTPUT_LIB_SVM_H_

#include <Eigen/Core>
#include <iostream>

namespace iNeural
{
    namespace LibSVM
    {
        int load(Eigen::MatrixXd &in, Eigen::MatrixXd &out, const char *filename, int min_inputs = 0);
        int load(Eigen::MatrixXd &in, Eigen::MatrixXd &out, std::istream &stream, int min_inputs = 0);

        void save(const Eigen::MatrixXd &in, const Eigen::MatrixXd &out, const char *filename);
        void save(const Eigen::MatrixXd &in, const Eigen::MatrixXd &out, std::ostream &stream);
    } // namespace LibSVM
} // namespace

#endif