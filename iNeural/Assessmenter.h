#ifndef INEURAL_ASSESSMENTER_H_
#define INEURAL_ASSESSMENTER_H_

#include <iNeural/input_output/Logger.h>

class Stopwatch;

namespace iNeural
{
    class Learner;
    class DataSet;

    class Assessmenter
    {
    public:
        virtual ~Assessmenter() {}
        virtual void assessment(Learner &learner, DataSet &dataSet) = 0;
    };

    class MulticlassAssessmenter : public Assessmenter
    {
        int interval;
        Logger *logger;
        Stopwatch *stopwatch;
        int iteration;

    public:
        MulticlassAssessmenter(int interval = 1, Logger::Target target = Logger::CONSOLE);
        virtual ~MulticlassAssessmenter();
        virtual void assessment(Learner &learner, DataSet &dataSet);
    };
} // namespace

#endif