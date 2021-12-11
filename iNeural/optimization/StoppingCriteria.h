#ifndef INEURAL_OPTIMIZATION_STOPPINGCRITERIA_H_
#define INEURAL_OPTIMIZATION_STOPPINGCRITERIA_H_

namespace iNeural
{
    class StoppingCriteria
    {
    public:
        static StoppingCriteria defaultValue;
        int maximalFunctionEvaluations;
        int maximalIterations;
        int maximalRestarts;
        double minimalValue;
        double minimalValueDifferences;
        double minimalSearchSpaceStep;

        StoppingCriteria();
    };
} // namespace

#endif