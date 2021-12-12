#ifndef INEURAL_UTIL_ASSERTION_MACROS_H_
#define INEURAL_UTIL_ASSERTION_MACROS_H_

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

#define INEURAL_CHECK(x) assert(x)

#define INEURAL_CHECK_EQUALS(x, y) \
    {                              \
        if (!((x) == (y)))         \
        {                          \
            INEURAL_TRACE((x));    \
            INEURAL_TRACE("!=");   \
            INEURAL_TRACE((y));    \
        }                          \
        assert((x) == (y));        \
    }

#define INEURAL_CHECK_NOT_EQUALS(x, y) \
    assert((x) != (y))

#define INEURAL_CHECK_EQUALS_DELTA(x, y, delta) \
    assert(iNeural::equals((x), (y), (delta)))

#define INEURAL_CHECK_WITHIN(x, min, max) \
    assert(((min) <= (x)) && ((x) <= (max)));

#define INEURAL_CHECK_NAN(value)               \
    if (iNeural::isNaN(value))                 \
    {                                          \
        INEURAL_TRACE("nan");                  \
        INEURAL_CHECK(!iNeural::isNaN(value)); \
    }

#define INEURAL_CHECK_INF(value)               \
    if (iNeural::isInf(value))                 \
    {                                          \
        INEURAL_TRACE("inf");                  \
        INEURAL_CHECK(!iNeural::isInf(value)); \
    }

#define INEURAL_CHECK_INF_AND_NAN(value) \
    INEURAL_CHECK_INF(value);            \
    INEURAL_CHECK_NAN(value);

#else // NDEBUG
#define INEURAL_CHECK(x)
#define INEURAL_CHECK_EQUALS(x, y)
#define INEURAL_CHECK_NOT_EQUALS(x, y)
#define INEURAL_CHECK_EQUALS_DELTA(x, y, delta)
#define INEURAL_CHECK_WITHIN(x, min, max)
#define INEURAL_CHECK_NAN(value)
#define INEURAL_CHECK_INF(value)
#define INEURAL_CHECK_INF_AND_NAN(value)
#endif // NDEBUG

#endif