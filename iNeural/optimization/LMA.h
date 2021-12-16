#ifndef INEURAL_OPTIMIZATION_LMA_H_
#define INEURAL_OPTIMIZATION_LMA_H_

#include <iNeural/optimization/Optimizer.h>
#include <iNeural/optimization/StoppingCriteria.h>
#include <Eigen/Core>
#include <optimization.h>

namespace iNeural
{

    class LMA : public Optimizer
    {
        StoppingCriteria stop;
        Optimizable *opt; // DON'T DELETE
        Eigen::VectorXd optimum;
        int iteration, n;

        alglib_impl::ae_state envState;

        Eigen::VectorXd parameters, errorValues, gradient;

        alglib::real_1d_array xIn;
        alglib::minlmstate state;

    public:
        LMA();
        virtual ~LMA();
        virtual void setOptimizable(Optimizable &opt);
        virtual void setStopCriteria(const StoppingCriteria &stop);
        virtual void optimize();
        virtual bool step();
        virtual Eigen::VectorXd result();
        virtual std::string name();

    private:
        void initialize();
        void reset();
    };

} // namespace

#endif
