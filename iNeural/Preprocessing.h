#ifndef INEURAL_PREPROCESSING_H_
#define INEURAL_PREPROCESSING_H_

#include <Eigen/Core>

namespace iNeural
{
    void scaleData(Eigen::MatrixXd &data, double min = -1.0, double max = 1.0);

    void filter(const Eigen::MatrixXd &x, Eigen::MatrixXd &y, const Eigen::MatrixXd &b, const Eigen::MatrixXd &a);
    void downsample(const Eigen::MatrixXd &y, Eigen::MatrixXd &d, int downSamplingFactor);
    Eigen::MatrixXd sampleRandomPatches(const Eigen::MatrixXd &images, int channels, int rows, int cols, int samples, int patchRows, int patchCols);
} // namespace

#endif