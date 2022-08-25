#ifndef INEURAL_UTIL_RANDOM_H_
#define INEURAL_UTIL_RANDOM_H_

#include <iNeural/util/AssertionMacros.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>

namespace iNeural
{
    class RandomNumberGenerator
    {
    private:
        mutable std::mt19937 m_generator;

    public:
        RandomNumberGenerator();

        void seed(unsigned int seed);
        int generateInt(int min, int range) const;
        size_t generateIndex(size_t size) const;

        template <class T>
        T generate(T min, T range) const
        {
            OPENANN_CHECK(range >= T());
            if (range == T())
                return min;
            else
                return (T)rand() / (T)RAND_MAX * range + min;
        }

        template <class T>
        T sampleNormalDistribution() const
        {
            return std::sqrt(T(-2) * std::log(generate(T(), T(1)))) *
                   std::cos(T(2) * T(M_PI) * generate(T(), T(1)));
        }

        template <class C>
        void generateIndices(int n, C &result, bool initialized = false)
        {
            if (!initialized)
            {
                OPENANN_CHECK_EQUALS(result.size(), 0);
                for (int i = 0; i < n; i++)
                    result.push_back(i);
            }
            else
            {
                OPENANN_CHECK_EQUALS(result.size(), (size_t)n);
            }
            std::shuffle(result.begin(), result.end(), m_generator);
        }

        template <class M>
        void fillNormalDistribution(M &matrix, double stdDev = 1.0)
        {
            const double *end = matrix.data() + matrix.rows() * matrix.cols();
            for (double *p = matrix.data(); p < end; p++)
                *p = sampleNormalDistribution<double>() * stdDev;
        }
    };
}

#endif