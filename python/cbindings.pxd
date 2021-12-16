from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "<iostream>" namespace "std":
  cdef cppclass ostream
  ostream& write "operator<<" (ostream& os, char* str)


cdef extern from "Eigen/Dense" namespace "Eigen":
  cdef cppclass VectorXd:
    VectorXd()
    VectorXd(int rows)
    VectorXd(VectorXd&)
    double* data()
    int rows()
    double& get "operator()"(int rows)

  cdef cppclass MatrixXd:
    MatrixXd()
    MatrixXd(int rows, int cols)
    double& coeff(int row, int col)
    double* data()
    int rows()
    int cols()
    double& get "operator()"(int rows, int cols)

  cdef cppclass MatrixXi:
    MatrixXi()
    MatrixXi(int rows, int cols)
    int& coeff(int row, int col)
    int* data()
    int rows()
    int cols()
    int& get "operator()"(int rows, int cols)


cdef extern from "iNeural/iNeural" namespace "iNeural::iNeuralLibraryInfo":
  char* VERSION
  char* URL
  char* DESCRIPTION
  char* COMPILATION_TIME
  char* COMPILER_FLAGS

cdef extern from "iNeural/iNeural" namespace "iNeural":
  void useAllCores()


cdef extern from "iNeural/ActivationFuncs.h" namespace "iNeural":
  cdef enum ActivationFunc:
    LOGISTIC
    TANH
    TANH_SCALED
    RECTIFIER
    LINEAR
    SOFTMAX

cdef extern from "iNeural/Network.h" namespace "iNeural":
  cdef enum ErrorFunc:
    NO_E_DEFINED
    MSE
    CE


cdef extern from "iNeural/input_output/Logger.h" namespace "iNeural::Logger":
  cdef enum Target:
    NONE
    CONSOLE
    FILE
    APPEND_FILE

cdef extern from "iNeural/input_output/Logger.h" namespace "iNeural::Log":
  cdef enum LogLevel:
    DISABLED
    ERROR
    INFO
    DEBUG

cdef extern from "iNeural/input_output/Logger.h" namespace "iNeural":
  cdef cppclass Log:
    Log()
    ostream& get(LogLevel level, char* namespace)

cdef extern from "iNeural/input_output/Logger.h" namespace "iNeural::Log":
  void setDisabled()
  void setError()
  void setInfo()
  void setDebug()


cdef extern from "iNeural/util/Random.h" namespace "iNeural":
  cdef cppclass RandomNumberGenerator:
    void seed(unsigned int seed)


cdef extern from "iNeural/layers/Layer.h" namespace "iNeural":
  cdef cppclass OutputInfo:
    bool bias
    vector[int] dimensions
    int outputs()

  cdef cppclass Layer:
    OutputInfo initialize(vector[double*]& param, vector[double*] derivative)
    void initializeParameters()
    void updatedParameters()
    void forwardPropagate(VectorXd* x, VectorXd*& y, bool dropout)
    void backpropagate(VectorXd* ein, VectorXd*& eout)
    MatrixXd& getOutput()
    VectorXd getParameters()


cdef extern from "iNeural/Regularization.h" namespace "iNeural":
  cdef cppclass Regularization:
    double l1Penalty
    double l2Penalty
    double maxSquaredWeightNorm

    Regularization(double l1Penalty, double l2Penalty,
                   double maxSquaredWeightNorm)


cdef extern from "iNeural/layers/SigmaPi.h" namespace "iNeural::SigmaPi":
  cdef cppclass Constraint:
    double constrain "operator()" (int p1, int p2)
    double constrain "operator()" (int p1, int p2, int p3)
    double constrain "operator()" (int p1, int p2, int p3, int p4)

cdef extern from "iNeural/layers/SigmaPi.h" namespace "iNeural":
  cdef cppclass SigmaPi(Layer):
    SigmaPi(OutputInfo info, bool bias, ActivationFunc act, double stdDev)
    SigmaPi& secondOrderNodes(int numbers)
    SigmaPi& thirdOrderNodes(int numbers)
    SigmaPi& fourthOrderNodes(int numbers)
    SigmaPi& secondOrderNodes(int numbers, Constraint& constrain)
    SigmaPi& thirdOrderNodes(int numbers, Constraint& constrain)
    SigmaPi& fourthOrderNodes(int numbers, Constraint& constrain)

cdef extern from "iNeural/layers/SigmaPiConstraints.h" namespace "iNeural":
  cdef cppclass DistanceConstraint(Constraint):
    DistanceConstraint(long width, long height)
  cdef cppclass SlopeConstraint(Constraint):
    SlopeConstraint(long width, long height)
  cdef cppclass TriangleConstraint(Constraint):
    TriangleConstraint(long width, long height, double resolution)


cdef extern from "iNeural/input_output/DataStream.h" namespace "iNeural":
  cdef cppclass DataStream:
    DataStream(int cacheSize)
    DataStream& setLearner(Learner& learner)
    DataStream& setOptimizer(Optimizer& opt)
    void addSample(VectorXd* x, VectorXd* t)


cdef extern from "iNeural/input_output/DataSet.h" namespace "iNeural":
  cdef cppclass DataSet:
    int samples()
    int inputs()
    int outputs()
    VectorXd& getInstance(int i)
    VectorXd& getTarget(int i)
    void finishIteration(Learner& learner)

cdef extern from "iNeural/input_output/DirectStorageDataSet.h" namespace "iNeural":
  cdef cppclass DirectStorageDataSet(DataSet):
    DirectStorageDataSet(MatrixXd* input, MatrixXd* output)

cdef extern from "iNeural/input_output/LibSVM.h":
  int libsvm_load "iNeural::LibSVM::load" (MatrixXd& input, MatrixXd& output,
                                           char *filename, int min_inputs)
  void save (MatrixXd& input, MatrixXd& output, char *filename)


cdef extern from "iNeural/optimization/StoppingCriteria.h" namespace "iNeural":
  cdef cppclass StoppingCriteria:
    StoppingCriteria()
    int maximalFunctionEvaluations
    int maximalIterations
    int maximalRestarts
    double minimalValue
    double minimalValueDifferences
    double minimalSearchSpaceStep


cdef extern from "iNeural/optimization/Optimizable.h" namespace "iNeural":
  cdef cppclass Optimizable:
    bool providesInitialization()
    void initialize()
    VectorXd& currentParameters()
    void setParameters(VectorXd& parameters)
    int dimension()
    double error()
    double error_from "error" (unsigned int i)
    bool providesGradient()
    VectorXd gradient_from "gradient" (unsigned int i)
    VectorXd gradient()


cdef extern from "iNeural/optimization/Optimizer.h" namespace "iNeural":
  cdef cppclass Optimizer:
    void setOptimizable(Optimizable& optimizable)
    void setStopCriteria(StoppingCriteria& sc)
    void optimize()
    VectorXd result()
    bool step()
    string name()


cdef extern from "iNeural/optimization/MiniBSGD.h" namespace "iNeural":
  cdef cppclass MBSGD(Optimizer):
    MBSGD(double learningRate, double momentum, int batchSize, bool nesterov,
       double learningRateDecay, double minimalLearningRate, 
       double momentumGain, double maximalMomentum,
       double minGain, double maxGain)

cdef extern from "iNeural/optimization/LMA.h" namespace "iNeural":
  cdef cppclass LMA(Optimizer):
    LMA()

cdef extern from "iNeural/optimization/CG.h" namespace "iNeural":
  cdef cppclass CG(Optimizer):
    CG()

cdef extern from "iNeural/optimization/LBFGS.h" namespace "iNeural":
  cdef cppclass LBFGS(Optimizer):
    LBFGS(int m)


cdef extern from "iNeural/Learner.h" namespace "iNeural":
  cdef cppclass Learner(Optimizable):
    Learner& trainingSet(MatrixXd& input, MatrixXd& output)
    Learner& trainingSet(DataSet& dataset)
    MatrixXd predict "operator()" (MatrixXd& x)

cdef extern from "iNeural/Network.h" namespace "iNeural":
  cdef cppclass Net(Learner):
    Network()
    Network& inputLayer(int dim1, int dim2, int dim3)
    Network& alphaBetaFilterLayer(double deltaT, double stdDev)
    Network& fullyConnectedLayer(int units, ActivationFunc act, double stdDev,
                             bool bias)
    Network& restrictedBoltzmannMachineLayer(int H, int cdN, double stdDev,
                                         bool backprop)
    Network& compressedLayer(int units, int params, ActivationFunc act,
                         string compression, double stdDev, bool bias)
    Network& extremeLayer(int units, ActivationFunc act, double stdDev,
                      bool bias)
    Network& intrinsicPlasticityLayer(double targetMean, double stdDev)
    Network& convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                            ActivationFunc act, double stdDev, bool bias)
    Network& subsamplingLayer(int kernelRows, int kernelCols,
                          ActivationFunc act, double stdDev, bool bias)
    Network& maxPoolingLayer(int kernelRows, int kernelCols)
    Network& localReponseNormalizationLayer(double k, int n, double alpha,
                                        double beta)
    Network& dropoutLayer(double dropoutProbability)
    Network& outputLayer(int units, ActivationFunc act, double stdDev)
    Network& compressedOutputLayer(int units, int params, ActivationFunc act,
                               string& compression, double stdDev)
    Network& addLayer(Layer *layer)
    Network& addOutputLayer(Layer *layer)

    Network& setRegularization(double l1Penalty, double l2Penalty,
                           double maxSquaredWeightNorm)
    Network& setErrorFunc(ErrorFunc errorFunc)
    Network& useDropout(bool activate)

    unsigned int numberOflayers()
    Layer& getLayer(unsigned int l)
    OutputInfo getOutputInfo(int l)
    DataSet* propagateDataSet(DataSet& dataSet, int l)

    void save(string& fileName)
    void load(string& fileName)

cdef extern from "iNeural/RBM.h" namespace "iNeural":
  cdef cppclass RBM(Learner):
    RBM(int D, int H, int cdN, double stdDev, bool backprop,
        Regularization regularization)
    int visibleUnits()
    int hiddenUnits()
    MatrixXd& getWeights()
    MatrixXd& getVisibleProbs()
    MatrixXd& getVisibleSample()
    MatrixXd reconstructProb(int n, int steps)
    void sampleHgivenV()
    void sampleVgivenH()

cdef extern from "iNeural/SparseAutoEncoder.h" namespace "iNeural":
  cdef cppclass SparseAutoEncoder(Learner):
    SparseAutoEncoder(int D, int H, double beta, double rho, double lmbda, ActivationFunc act)
    MatrixXd getInputWeights()
    MatrixXd getOutputWeights()
    VectorXd reconstruct(VectorXd& x)

cdef extern from "iNeural/Transformer.h" namespace "iNeural":
  cdef cppclass Transformer:
    Transformer& fit(MatrixXd& X)
    Transformer& fitPartial(MatrixXd& X)
    MatrixXd transform(MatrixXd& X)

cdef extern from "iNeural/Normalization.h" namespace "iNeural":
  cdef cppclass Normalization(Transformer):
    Normalization()
    MatrixXd getMean()
    MatrixXd getStd()

cdef extern from "iNeural/PCA.h" namespace "iNeural":
  cdef cppclass PCA(Transformer):
    PCA(int components, bool whiten)
    VectorXd explainedVarianceRatio()

cdef extern from "iNeural/ZeroCAWhitening.h" namespace "iNeural":
  cdef cppclass ZeroCAWhitening(Transformer):
    ZeroCAWhitening()

cdef extern from "iNeural/KMeans.h" namespace "iNeural":
  cdef cppclass KMeans(Transformer):
    KMeans(int D, int K)
    MatrixXd getCenters()

cdef extern from "iNeural/CompressionMatrixFactory.h" namespace "iNeural::CompressionMatrixFactory":
  cdef enum Transformation:
    DCT
    GAUSSIAN
    SPARSE_RANDOM
    AVERAGE
    EDGE

cdef extern from "iNeural/Compressor.h" namespace "iNeural":
  cdef cppclass Compressor(Transformer):
    Compressor(int inputDim, int outputDim,
               Transformation transformation)
    int getOutputs()


cdef extern from "iNeural/Preprocessing.h" namespace "iNeural":
  MatrixXd sampleRandomPatches(MatrixXd& images, int channels, int rows,
                               int cols, int samples, int patchRows,
                               int patchCols)


cdef extern from "iNeural/Assessment.h" namespace "iNeural":
  double sse(Learner& learner, DataSet& dataSet)
  double mse(Learner& learner, DataSet& dataSet)
  double rmse(Learner& learner, DataSet& dataSet)
  double accuracy(Learner& learner, DataSet& dataSet)
  MatrixXi confusionMatrix(Learner& learner, DataSet& dataSet)
  int classificationHits(Learner& learner, DataSet& dataSet)
  double crossValidation(int folds, Learner& learner, DataSet& dataSet,
                         Optimizer& opt)


cdef extern from "iNeural/CollectiveLearner.h" namespace "iNeural":
  cdef cppclass CollectiveLearner:
    CollectiveLearner& addLearner(Learner& learner)
    CollectiveLearner& setOptimizer(Optimizer& optimizer)
    CollectiveLearner& train(DataSet& dataSet)
    MatrixXd predict "operator()" (MatrixXd& x)

cdef extern from "iNeural/Adaraise.h" namespace "iNeural":
  cdef cppclass AdaRaise(CollectiveLearner):
    AdaRaise()
    VectorXd getWeights()

cdef extern from "iNeural/Packaging.h" namespace "iNeural":
  cdef cppclass Packaging(EnsembleLearner):
    Packaging(double bagSize)