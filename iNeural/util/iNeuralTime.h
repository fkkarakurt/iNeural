#ifndef INEURAL_TIME_H
#define INEURAL_TIME_H

#ifdef _WIN32
#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>

int gettimeofday(struct timeval *t, void *timezone);

#define __need_clock_t
#include <time.h>

struct tms
{
    clock_t tms_utime;
    clock_t tms_stime;

    clock_t tms_cutime;
    clock_t tms_cstime;
};

clock_t times(struct tms *__buffer);

typedef long long suseconds_t;

#endif
#endif