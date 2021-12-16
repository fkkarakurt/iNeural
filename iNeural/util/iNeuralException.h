#ifndef INEURAL_UTIL_OPENANN_EXCEPTION_H_
#define INEURAL_UTIL_OPENANN_EXCEPTION_H_

#include <stdexcept>

namespace iNeural
{

    class iNeuralException : public std::logic_error
    {
    public:
        iNeuralException(const std::string &msg);
    };

} // namespace

#endif