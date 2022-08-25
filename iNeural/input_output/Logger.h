#ifndef INEURAL_INPUT_OUTPUT_LOGGER_H_
#define INEURAL_INPUT_OUTPUT_LOGGER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#ifndef NDEBUG
#define INEURAL_OUTPUT(msg) std::cout << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;
#define INEURAL_TRACE(msg) std::cerr << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;

#else
#define INEURAL_OUTPUT(msg) std::cout << msg << std::endl;
#define INEURAL_TRACE(msg)
#endif

#ifndef INEURAL_LOG_NAMESPACE
#define INEURAL_LOG_NAMESPACE NULL
#endif

#ifndef INEURAL_LOGLEVEL
#define INEURAL_LOGLEVEL iNeural::Log::DEBUG
#endif

#define INEURAL_LOG(level)                     \
    if (level > INEURAL_LOGLEVEL)              \
        ;                                      \
    else if (level > iNeural::Log::getLevel()) \
        ;                                      \
    else                                       \
        iNeural::Log().get(level, INEURAL_LOG_NAMESPACE)

#define INEURAL_DEBUG INEURAL_LOG(iNeural::Log::DEBUG)
#define INEURAL_INFO INEURAL_LOG(iNeural::Log::INFO)
#define INEURAL_ERROR INEURAL_LOG(iNeural::Log::ERROR)

namespace iNeural
{
    struct FloatingPointFormatter
    {
        double value;
        int precision;
        FloatingPointFormatter(double value, int precision) : value(value), precision(precision) {}
    };

    class Log
    {
    public:
        enum LogLevel
        {
            DISABLED = 0,
            ERROR,
            INFO,
            DEBUG
        };

        Log();
        virtual ~Log();

        std::ostream &get(LogLevel level, const char *name_space);

        static void setStream(std::ostream &stream);
        static std::ostream &getStream();
        static LogLevel &getLevel();

        static void setDisabled();
        static void setError();
        static void setInfo();
        static void setDebug();

    private:
        static std::ostream *stream;
        std::ostringstream message;
        LogLevel level;
    };

    class Logger
    {
    public:
        static bool deactivate;
        enum Target
        {
            NONE,
            CONSOLE,
            FILE,
            APPEND_FILE
        } target;

        std::string name;
        std::ofstream file;

        Logger(Target target, std::string name = "Logger");
        ~Logger();

        bool isActive();
    };

    std::ostream &operator<<(std::ostream &os, const FloatingPointFormatter &t);
    Logger &operator<<(Logger &logger, const FloatingPointFormatter &t);
    template <typename T>

    Logger &operator<<(Logger &logger, const T &t)
    {
        switch (logger.target)
        {
        case Logger::CONSOLE:
            std::cout << t << std::flush;
            break;
        case Logger::APPEND_FILE:
        case Logger::FILE:
            logger.file << t << std::flush;
            break;
        default:
            break;
        }
        return logger;
    }
}

#endif