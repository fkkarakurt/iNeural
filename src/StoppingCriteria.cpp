#include <iNeural/optimization/StoppingCriteria.h>
#include <limits>

namespace iNeural
{
    StoppingCriteria StoppingCriteria::defaultValue;

    StoppingCriteria::StoppingCriteria()
        : maximalFunctionEvaluations(-1),
          maximalIterations(-1),
          maximalRestarts(0),
          minimalValue(-std::numeric_limits<double>::max()),
          minimalValueDifferences(0),
          minimalSearchSpaceStep(0) {}
} // namespace