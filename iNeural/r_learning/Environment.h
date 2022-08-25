#ifndef INEURAL_R_LEARNING_ENVIRONMENT_H_
#define INEURAL_R_LEARNING_ENVIRONMENT_H_

#include <iNeural/r_learning/ActionSpace.h>
#include <iNeural/r_learning/StateSpace.h>
#include <Eigen/Core>

namespace iNeural
{
    class Environment : public StateSpace, public ActionSpace
    {
    public:
        virtual ~Environment() {}
        virtual void restart() = 0;
        virtual const State &getState() const = 0;
        virtual const Action &getAction() const = 0;
        virtual void stateTransition(const Action &action) = 0;
        virtual double reward() const = 0;
        virtual bool terminalState() const = 0;
        virtual bool successful() const = 0;
        virtual int stepsInEpisode() const = 0;
        virtual double deltaT() const { return 1.0; }
    };

}

#endif