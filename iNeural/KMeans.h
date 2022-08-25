#ifndef INEURAL_KMEANS_H_
#define INEURAL_KMEANS_H_

#include <iNeural/util/Random.h>
#include <iNeural/Transformer.h>
#include <Eigen/Core>
#include <vector>

namespace iNeural
{

    class KMeans : public Transformer
    {
        const int D;
        const int K;
        Eigen::MatrixXd C;
        Eigen::VectorXi v;
        bool initialized;
        RandomNumberGenerator rng;
        std::vector<int> clusterIndices;

    public:
        KMeans(int D, int K);

        virtual Transformer &fit(const Eigen::MatrixXd &X);
        virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X)
        {
            return (*this)(X);
        }

        Eigen::MatrixXd operator()(const Eigen::MatrixXd &X);

        Eigen::MatrixXd getCenters();

    private:
        void initialize(const Eigen::MatrixXd &X);
        void findClusters(const Eigen::MatrixXd &X);
        void updateCenters(const Eigen::MatrixXd &X);
    };

} // namespace
#endif