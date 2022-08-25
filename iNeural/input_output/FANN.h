#ifndef INEURAL_INPUT_OUTPUT_FANN_H_
#define INEURAL_INPUT_OUTPUT_FANN_H_

#include <Eigen/Core>
#include <iostream>

namespace iNeural
{
    namespace FANN
    {

        int load(Eigen::MatrixXd &in, Eigen::MatrixXd &out, const char *filename);
        int load(Eigen::MatrixXd &in, Eigen::MatrixXd &out, std::istream &stream);
        void save(const Eigen::MatrixXd &in, const Eigen::MatrixXd &out, const char *filename);
        void save(const Eigen::MatrixXd &in, const Eigen::MatrixXd &out, std::ostream &stream);
    } // namespace FANN
} // namespace

#endif