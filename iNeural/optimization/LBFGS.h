#ifndef INEURAL_OPTIMIZATION_LBFGS_H_
#define INEURAL_OPTIMIZATION_LBFGS_H_

#include <iNeural/optimization/Optimizer.h>
#include <iNeural/optimization/StoppingCriteria.h>
#include <Eigen/Core>
#include <optimization.h>

namespace iNeural
{

    class LBFGS : public Optimizer
    {
        StoppingCriteria stop;
        Optimizable *opt; // DON'T DELETE
        Eigen::VectorXd optimum;
        int iteration, n, m;
        Eigen::VectorXd parameters, gradient;
        double error;

        alglib_impl::ae_state envState;
        alglib::minlbfgsstate state;
        alglib::real_1d_array xIn;

    public:
        LBFGS(int m = 10);
        virtual ~LBFGS() {}
        virtual void setStopCriteria(const StoppingCriteria &stop);
        virtual void setOptimizable(Optimizable &optimizable);
        virtual void optimize();
        virtual bool step();
        void initialize();
        void reset();
        virtual Eigen::VectorXd result();
        virtual std::string name();
    };

} // namespace

#endif