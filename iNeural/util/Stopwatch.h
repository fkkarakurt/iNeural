#ifndef INEURAL_UTIL_STOPWATCH_H_
#define INEURAL_UTIL_STOPWATCH_H_

#include <iNeuralTime.h>

class Stopwatch
{
    unsigned long begin, duration;

    inline unsigned long getSystime()
    {
        timeval t;
        gettimeofday(&t, 0);
        return (unsigned long)t.tv_sec * 1000000L + (unsigned long)t.tv_usec;
    }

public:
    enum Precision
    {
        MICROSECOND,
        MILLISECOND,
        HUNDREDTHS,
        TENTHS,
        SECONDS
    };

    Stopwatch() : begin(getSystime()), duration(0){};

    void start()
    {
        begin = getSystime();
    }

    inline unsigned long stop()
    {
        duration = getSystime() - begin;
        return duration;
    }

    inline unsigned long stop(Precision p)
    {
        unsigned long duration = stop();
        switch (p)
        {
        case SECONDS:
            return duration / 1000000L;
        case TENTHS:
            return duration / 100000L;
        case HUNDREDTHS:
            return duration / 10000L;
        case MILLISECOND:
            return duration / 1000L;
        case MICROSECOND:
        default:
            return duration;
        }
    }

    inline unsigned long getDuration()
    {
        return duration;
    }
};

#endif