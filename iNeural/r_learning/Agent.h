#ifndef INEURAL_R_LEARNING_AGENT_H_
#define INEURAL_R_LEARNING_AGENT_H_

#include <iNeural/r_learning/Environment.h>

namespace iNeural
{
    class Agent
    {
    public:
        virtual ~Agent() {}
        virtual void abandoneIn(Environment &environment) = 0;
        virtual void chooseAction() = 0;
        virtual void chooseOptimalAction() = 0;
    };

}

#endif