#ifndef INEURAL_EASE_H_
#define INEURAL_EASE_H_

#include <iNeural/Network.h>
#include <string>

namespace iNeural
{
    class StoppingCriteria;

    void train(Network &net, std::string algorithm, ErrorFunc errorFunction, const StoppingCriteria &stop, bool reinitialize = false, bool dropout = false);
    void makeMLNN(Network &net, ActivationFunc g, ActivationFunc h, int D, int F, int H, ...);

} // namespace

#endif