#include "iNeural/layers/Layer.h"
#include <numeric>

namespace iNeural
{
    int OutputInfo::outputs()
    {
        return std::accumulate(dimensions.begin(),
                               dimensions.end(),
                               1,
                               std::multiplies<int>());
    }
}