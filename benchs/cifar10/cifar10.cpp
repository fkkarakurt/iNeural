#include <iNeural/iNeural>
#include <iNeural/optimization/MiniBSGD.h>
#include <iNeural/input_output/DirectStorageDataSet.h>
#include <iNeural/Assessmenter.h>
#include "CIFARLoader.h"

int main(int argc, char **argv)
{
    iNeural::useAllCores();

    std::string directory = ".";
    if (argc > 1)
        directory = std::string(argv[1]);

    CIFARLoader loader(directory);

    iNeural::Network net;                                         // Nodes per layer:
    net.inputLayer(loader.C, loader.X, loader.Y)                  //   3 x 32 x 32
        .convolutionalLayer(50, 5, 5, iNeural::RECTIFIER, 0.05)   //  50 x 28 x 28
        .maxPoolingLayer(2, 2)                                    //  50 x 14 x 14
        .convolutionalLayer(30, 3, 3, iNeural::RECTIFIER, 0.05)   //  30 x 12 x 12
        .maxPoolingLayer(2, 2)                                    //  30 x  6 x  6
        .convolutionalLayer(20, 3, 3, iNeural::RECTIFIER, 0.05)   //  20 x  4 x  4
        .maxPoolingLayer(2, 2)                                    //  20 x  2 x  2
        .fullyConnectedLayer(100, iNeural::RECTIFIER, 0.05, true) // 100
        .fullyConnectedLayer(50, iNeural::RECTIFIER, 0.05, true)  //  50
        .outputLayer(loader.F, iNeural::SOFTMAX, 0.05)            //  10
        .trainingSet(loader.trainingInput, loader.trainingOutput);
    iNeural::MulticlassAssessmenter assessmenter(1, iNeural::Logger::FILE);
    iNeural::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput, &assessmenter); //?????????????????????//
    net.validationSet(testSet);
    net.setErrorFunc(iNeural::CE);
    INEURAL_INFO << "Created MLP.";
    INEURAL_INFO << "D = " << loader.D << ", F = " << loader.F << ", N = "
                 << loader.trainingN << ", L = " << net.dimension();

    iNeural::StoppingCriteria stop;
    stop.maximalIterations = 100;
    iNeural::MiniBSGD optimizer(0.01, 0.6, 10, false, 1.0, 0.0, 0.0, 1.0, 0.01, 100.0);
    optimizer.setOptimizable(net);
    optimizer.setStopCriteria(stop);
    optimizer.optimize();

    INEURAL_INFO << "Error = " << net.error();
    INEURAL_INFO << "Wrote data to evaluation-*.log.";

    return EXIT_SUCCESS;
}