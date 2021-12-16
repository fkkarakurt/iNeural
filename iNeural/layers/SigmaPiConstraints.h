#ifndef INEURAL_LAYERS_SIGMA_PI_CONSTRAINTS_H_
#define INEURAL_LAYERS_SIGMA_PI_CONSTRAINTS_H_

#include <iNeural/util/AssertionMacros.h>
#include <iNeural/layers/SigmaPi.h>
#include <iostream>
#include <map>

namespace iNeural
{

    struct DistanceConstraint : public SigmaPi::Constraint
    {
        DistanceConstraint(size_t width, size_t height)
            : width(width), height(height)
        {
        }

        virtual ~DistanceConstraint() {}

        virtual double operator()(int p1, int p2) const
        {
            double x1 = p1 % width;
            double y1 = p1 / width;
            double x2 = p2 % width;
            double y2 = p2 / width;

            INEURAL_CHECK(y1 < height);
            INEURAL_CHECK(y2 < height);

            return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
        }

    private:
        size_t width;
        size_t height;
    };

    struct SlopeConstraint : public SigmaPi::Constraint
    {
        SlopeConstraint(size_t width, size_t height)
            : width(width), height(height)
        {
        }

        virtual ~SlopeConstraint() {}

        virtual double operator()(int p1, int p2) const
        {
            double x1 = p1 % width;
            double y1 = p1 / width;
            double x2 = p2 % width;
            double y2 = p2 / width;

            INEURAL_CHECK(y1 < height);
            INEURAL_CHECK(y2 < height);

            return (x1 == x2) ? (3.14159265358979323846 / 2) : std::atan((y2 - y1) / (x2 - x1));
        }

    private:
        size_t width;
        size_t height;
    };

    struct TriangleConstraint : public SigmaPi::Constraint
    {
        struct AngleTuple
        {
            AngleTuple(double a, double b, double c) : alpha(a), beta(b), gamma(c)
            {
            }

            bool operator<(const AngleTuple &tuple) const
            {
                if (fabs(alpha - tuple.alpha) > 0.001)
                    return alpha < tuple.alpha;
                else if (fabs(beta - tuple.beta) > 0.001)
                    return beta < tuple.beta;
                else
                    return fabs(gamma - tuple.gamma) > 0.001 && gamma < tuple.gamma;
            }

            double alpha;
            double beta;
            double gamma;
        };

        TriangleConstraint(size_t width, size_t height, double resolution = 3.14159265358979323846 / 4)
            : width(width), height(height), resolution(resolution)
        {
        }

        virtual ~TriangleConstraint() {}

        virtual double operator()(int p1, int p2, int p3) const
        {
            int nr = partition.size() / 3.0;

            int x1 = p1 % width;
            int y1 = p1 / width;

            int x2 = p2 % width;
            int y2 = p2 / width;

            int x3 = p3 % width;
            int y3 = p3 / width;

            if (x2 < x3)
            {
                std::swap(x2, x3);
                std::swap(y2, y3);
            }

            int as = (x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3);
            int bs = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
            int cs = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);

            double alpha = std::floor(std::acos((as - bs - cs) / (-2 * std::sqrt(bs * cs))) / resolution);
            double beta = std::floor(std::acos((bs - cs - as) / (-2 * std::sqrt(cs * as))) / resolution);
            double gamma = std::floor(std::acos((cs - as - bs) / (-2 * std::sqrt(as * bs))) / resolution);

            AngleTuple t1(alpha, beta, gamma);
            AngleTuple t2(beta, gamma, alpha);
            AngleTuple t3(gamma, alpha, beta);

            std::map<AngleTuple, int>::const_iterator it = partition.find(t1);

            std::map<AngleTuple, int> &p = const_cast<std::map<AngleTuple, int> &>(partition);

            if (it == partition.end())
            {
                p[t1] = nr;
                p[t2] = nr;
                p[t3] = nr;
            }
            else
            {
                return it->second;
            }

            return nr;
        }

    private:
        std::map<AngleTuple, int> partition;

        size_t width;
        size_t height;
        double resolution;
    };

} // namespace OpenANN

#endif