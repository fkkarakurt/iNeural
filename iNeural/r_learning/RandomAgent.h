#ifndef INEURAL_R_LEARNING_RANDOM_AGENT_H_
#define INEURAL_R_LEARNING_RANDOM_AGENT_H_

#include <iNeural/r_learning/Agent.h>

namespace iNeural
{
    class RandomAgent : public Agent
    {
        Environment *environment;
        double accumulatedReward;
        double lastReturn;

    public:
        RandomAgent();
        virtual void abandoneIn(Environment &environment);
        virtual void chooseAction();
        virtual void chooseOptimalAction();
    };

}

#endif