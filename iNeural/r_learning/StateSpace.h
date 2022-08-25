#ifndef INEURAL_R_LEARNING_STATE_SPACE_H_
#define INEURAL_R_LEARNING_STATE_SPACE_H_

#include <Eigen/Core>
#include <vector>

namespace iNeural
{
    class StateSpace
    {
    public:
        typedef Eigen::VectorXd State;
        typedef std::vector<State> S;
        virtual ~StateSpace() {}
        virtual int stateSpaceDimension() const = 0;
        virtual bool stateSpaceContinuous() const = 0;
        virtual int stateSpaceElements() const = 0;
        virtual const State &stateSpaceLowerBound() const = 0;
        virtual const State &stateSpaceUpperBound() const = 0;
        virtual const S &getDiscreteStateSpace() const = 0;
    };

}

#endif