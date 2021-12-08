#ifndef INEURAL_REGULARIZATION_H
#define INEURAL_REGULARIZATION_H_

namespace iNeural
{

    class Regularization
    {
    public:
        double l1penal;
        double l2penal;
        double maxSquaredWeightNorm;

        Regularization(double l1penal = 0.0, double l2penal = 0.0, double maxSquaredWeightNorm = 0.0) : l1penal(l1penal), l2penal(l2penal), maxSquaredWeightNorm(maxSquaredWeightNorm)
        {
        }
    };

} // namespace

#endif