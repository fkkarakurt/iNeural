#ifndef INEURAL_OPTIMIZATION_H_
#define INEURAL_OPTIMIZATION_H_

#include <iNeural/optimization/Optimizer.h>
#include <iNeural/optimization/StoppingCriteria.h>
#include <Eigen/Core>

namespace iNeural
{

    class CG : public Optimizer
    {
        StoppingCriteria stop;
        Optimizable *opt; // DON'T DELETE
        Eigen::VectorXd optimum;
        int iteration, n;
        Eigen::VectorXd parameters, gradient;
        double error;

        /* Error Code : 454
        alglib_impl::ae_state envState;
        alglib::mincgstate state;
        alglib::real_1d_array xIn;
        */

    public:
        CG();
        ~CG();
        virtual void setOptimizable(Optimizable &opt);
        virtual void setStopCriteria(const StoppingCriteria &stop);
        virtual bool step();
        virtual void optimize();
        virtual Eigen::VectorXd result();
        virtual std::string name();

    private:
        void initialize();
        void reset();
    };

} // namespace OpenANN

#endif

// ERROR CODE : 454 ==> algLib and Optimization libraries will be prepared