#ifndef INEURAL_OPTIMIZATION_OPTIMIZER_H_
#define INEURAL_OPTIMIZATION_OPTIMIZER_H_

#include <Eigen/Core>
#include <string>

namespace iNeural
{
    class Optimizable;
    class StoppingCriteria;

    class Optimizer
    {
    public:
        virtual ~Optimizer() {}
        virtual void setOptimizable(Optimizable &optimizable) = 0;
        virtual void setStopCriteria(const StoppingCriteria &sc) = 0;
        virtual void optimize() = 0;
        virtual bool step() = 0;
        virtual Eigen::VectorXd result() = 0;
        virtual std::string name() = 0;
    };
} // namespace

#endif