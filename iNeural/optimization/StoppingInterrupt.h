#ifndef INEURAL_OPTIMIZATION_STOPPING_INTERRIPT_H_
#define INEURAL_OPTIMIZATION_STOPPING_INTERRIPT_H_

namespace iNeural
{
    class StoppingInterrupt
    {
        static int observers;
        static bool stoppingInterruptSignal;

    public:
        StoppingInterrupt();

        ~StoppingInterrupt();

        bool isSignaled();

    private:
        static void setStoppingInterruptSignal(int param);
    };
} // namespace

#endif