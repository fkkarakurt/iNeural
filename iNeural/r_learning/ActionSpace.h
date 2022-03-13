#ifndef INEURAL_R_LEARNING_ACTION_SPACE_H_
#define INEURAL_R_LEARNING_ACTION_SPACE_H_

#include <Eigen/Core>
#include <vector>

namespace iNeural
{

    class ActionSpace
    {
    public:
        typedef Eigen::VectorXd Action;
        typedef std::vector<Action> A;
        virtual ~ActionSpace() {}
        virtual int actionSpaceDimension() const = 0;
        virtual bool actionSpaceContinuous() const = 0;
        virtual int actionSpaceElements() const = 0;
        virtual const Action &actionSpaceLowerBound() const = 0;
        virtual const Action &actionSpaceUpperBound() const = 0;
        virtual const A &getDiscreteActionSpace() const = 0;
    };

}

#endif