#ifndef INEURAL_OPTIMIZATION_OPTIMIZABLE_H_
#define INEURAL_OPTIMIZATION_OPTIMIZABLE_H_
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <Eigen/Core>
#include <vector>

namespace iNeural
{
    class Optimizable
    {
    public:
        virtual ~Optimizable() {}
        virtual bool providesInitialization() = 0;
        virtual void initialize() = 0;
        virtual unsigned dimension() = 0;
        virtual const Eigen::VectorXd &currentParameters() = 0;
        virtual void setParameters(const Eigen::VectorXd &parameters) = 0;
        virtual double error() = 0;
        virtual bool providesGradient() = 0;
        virtual Eigen::VectorXd gradient() = 0;
        virtual unsigned examples() { return 1; }
        virtual double error(unsigned n) { return error(); }
        virtual Eigen::VectorXd gradient(unsigned n) { return gradient(); }
        virtual void errorGradient(int n, double &value, Eigen::VectorXd &grad);
        virtual void errorGradient(double &value, Eigen::VectorXd &grad);
        virtual Eigen::VectorXd error(std::vector<int>::const_iterator startN, std::vector<int>::const_iterator endN);
        virtual Eigen::VectorXd gradient(std::vector<int>::const_iterator startN, std::vector<int>::const_iterator endN);
        virtual void errorGradient(std::vector<int>::const_iterator startN, std::vector<int>::const_iterator endN, double &value, Eigen::VectorXd &grad);
        virtual void finishedIteration() {}
    };
} // namespace

#endif