#ifndef INEURAL_NETWORK_H_
#define INEURAL_NETWORK_H_

#include <iNeural/Learner.h>
#include <iNeural/ActivationFuncs.h>
#include <iNeural/Regularization.h>
#include <iNeural/layers/Layer.h>
#include <vector>
#include <sstream>

namespace iNeural
{
    enum ErrorFunc
    {
        NO_E_DEFINED,
        MSE,
        CE
    };

    class Network : public Learner
    {
    protected:
        std::vector<OutputInfo> infos;
        std::vector<Layer *> layers;
        std::vector<double *> parameters;
        std::vector<double *> derivatives;
        Regularization regularization;
        ErrorFunc errorFunc;
        bool dropout;
        bool initialized;
        int P, L;
        Eigen::VectorXd parameterVector, tempGradient;
        Eigen::MatrixXd tempInput, tempOutput, tempError;
        std::stringstream architecture;

    public:
        Network();
        virtual ~Network();
        Network &inputLayer(int dim1, int dim2 = 1, int dim3 = 1);
        Network &alphaBetaFilterLayer(double deltaT, double stdDev = 0.05);
        Network &fullyConnectedLayer(int units, ActivationFunc act, double stdDev = 0.05, bool bias = true);
        Network &restrictedBoltzmannMachineLayer(int H, int cdN = 1, double stdDev = 0.01, bool backprop = true);
        Network &sparseAutoEncoderLayer(int H, double beta, double rho, ActivationFunc act);
        Network &compressedLayer(int units, int params, ActivationFunc act, const std::string &compression, double stdDev = 0.05, bool bias = true);
        Network &extremeLayer(int units, ActivationFunc act, double stdDev = 5.0, bool bias = true);
        Network &intrinsicPlasticityLayer(double targetMean, double stdDev = 1.0);
        Network &convolutionalLayer(int featureMaps, int kernelRows, int kernelCols, ActivationFunc act, double stdDev = 0.05, bool bias = true);
        Network &subsamplingLayer(int kernelRows, int kernelCols, ActivationFunc act, double stdDev = 0.05, bool bias = true);
        Network &maxPoolingLayer(int kernelRows, int kernelCols);
        Network &localReponseNormalizationLayer(double k, int n, double alpha, double beta);
        Network &dropoutLayer(double dropoutProbability);
        Network &outputLayer(int units, ActivationFunc act, double stdDev = 0.05, bool bias = true);
        Network &compressedOutputLayer(int units, int params, ActivationFunc act, const std::string &compression, double stdDev = 0.05, bool bias = true);
        Network &addLayer(Layer *layer);
        Network &addOutputLayer(Layer *layer);

        unsigned int numberOflayers();
        Layer &getLayer(unsigned int l);
        OutputInfo getOutputInfo(unsigned int l);
        DataSet *propagateDataSet(DataSet &dataSet, int l);

        void save(const std::string &fileName);
        void save(std::ostream &stream);
        void load(const std::string &fileName);
        void load(std::istream &stream);

        Network &setRegularization(double l1Penalty = 0.0, double l2Penalty = 0.0, double maxSquaredWeightNorm = 0.0);
        Network &setErrorFunc(ErrorFunc errorFunc);
        Network &useDropout(bool activate = true);

        virtual Eigen::VectorXd operator()(const Eigen::VectorXd &x);
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &X);
        virtual unsigned int dimension();
        virtual const Eigen::VectorXd &currentParameters();
        virtual void setParameters(const Eigen::VectorXd &parameters);
        virtual bool providesInitialization();
        virtual void initialize();
        virtual unsigned int examples();
        virtual double error(unsigned int n);
        virtual double error();
        virtual bool providesGradient();
        virtual Eigen::VectorXd gradient(unsigned int n);
        virtual Eigen::VectorXd gradient();
        virtual void errorGradient(int n, double &value, Eigen::VectorXd &grad);
        virtual void errorGradient(double &value, Eigen::VectorXd &grad);
        virtual void errorGradient(std::vector<int>::const_iterator startN, std::vector<int>::const_iterator endN, double &value, Eigen::VectorXd &grad);
        virtual void finishedIteration();

    protected:
        void initializeNetwork();
        void forwardPropagate(double *error);
        void backpropagate();
    };
}

#endif