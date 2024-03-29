#include <iNeural/util/Random.h>
#include <ctime>

namespace iNeural
{

    RandomNumberGenerator::RandomNumberGenerator()
    {
        static bool seedInitialized = false;
        if (!seedInitialized)
        {
            srand(std::time(0));
            seedInitialized = true;
        }
        m_generator.seed(std::time(0));
    }

    void RandomNumberGenerator::seed(unsigned int seed)
    {
        srand(seed);
        m_generator.seed(seed);
    }

    int RandomNumberGenerator::generateInt(int min, int range) const
    {
        INEURAL_CHECK(range >= 0);
        if (range == 0)
            return min;
        else
            return rand() % range + min;
    }

    size_t RandomNumberGenerator::generateIndex(size_t size) const
    {
        return (size_t)generateInt(0, size);
    }

}