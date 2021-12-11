#ifndef INUERAL_UTIL_ASSERTION_MACROS_H_
#define INUERAL_UTIL_ASSERTION_MACROS_H_

#ifndef NDEBUG

#include <iNeural/input_output/Logger.h>
#include <cassert>
#include <cmath>
#include <iostream>

namespace iNeural
{
    template <class T>
    bool equals(T a, T b, T delta)
    {
        return std::fabs(a - b) <= delta;
    }

    template <class T>
    bool isNaN(T value)
    {
        return std::isnan(value);
    }

    template <class T>
    bool isInf(T value)
    {
        return std::isinf(value);
    }
} // namespace

#define OPENANN_CHECK(x) assert(x)

#define OPENANN_CHECK_EQUALS(x, y) \
    {                              \
        if (!((x) == (y)))         \
        {                          \
            OPENANN_TRACE((x));    \
            OPENANN_TRACE("!=");   \
            OPENANN_TRACE((y));    \
        }                          \
        assert((x) == (y));        \
    }

#define OPENANN_CHECK_NOT_EQUALS(x, y) \
    assert((x) != (y))

#define OPENANN_CHECK_EQUALS_DELTA(x, y, delta) \
    assert(OpenANN::equals((x), (y), (delta)))

#define OPENANN_CHECK_WITHIN(x, min, max) \
    assert(((min) <= (x)) && ((x) <= (max)));

#define OPENANN_CHECK_NAN(value)               \
    if (OpenANN::isNaN(value))                 \
    {                                          \
        OPENANN_TRACE("nan");                  \
        OPENANN_CHECK(!OpenANN::isNaN(value)); \
    }

#define OPENANN_CHECK_INF(value)               \
    if (OpenANN::isInf(value))                 \
    {                                          \
        OPENANN_TRACE("inf");                  \
        OPENANN_CHECK(!OpenANN::isInf(value)); \
    }

#define OPENANN_CHECK_INF_AND_NAN(value) \
    OPENANN_CHECK_INF(value);            \
    OPENANN_CHECK_NAN(value);

#else // NDEBUG
#define OPENANN_CHECK(x)
#define OPENANN_CHECK_EQUALS(x, y)
#define OPENANN_CHECK_NOT_EQUALS(x, y)
#define OPENANN_CHECK_EQUALS_DELTA(x, y, delta)
#define OPENANN_CHECK_WITHIN(x, min, max)
#define OPENANN_CHECK_NAN(value)
#define OPENANN_CHECK_INF(value)
#define OPENANN_CHECK_INF_AND_NAN(value)
#endif // NDEBUG

#endif