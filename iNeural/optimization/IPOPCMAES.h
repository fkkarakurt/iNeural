#ifndef INEURAL_OPTIMIZATION_IPOPCMAES_H_
#define INEURAL_OPTIMIZATION_IPOPCMAES_H_

#include <iNeural/input_output/Logger.h>
#include <iNeural/optimization/StoppingCriteria.h>
#include <iNeural/optimization/Optimizer.h>
#include <Eigen/Core>

template<typename T> class CMAES;
template<typename T> class Parameters;

namespace iNeural
{

class IPOPCMAES : public Optimizer
{
  Logger logger;
  StoppingCriteria stop;
  bool maxFunEvalsActive;
  Optimizable* opt; // DONT DELETE
  CMAES<double>* cmaes;
  Parameters<double>* parameters;

  int currentIndividual;
  double* initialX;
  double* initialStdDev;
  double* const* population;
  double* fitnessValues;
  int restarts;
  int evaluations;
  int evaluationsAfterRestart;
  bool stopped;

  Eigen::VectorXd optimum;
  double optimumValue;

  double sigma0;

public:

  IPOPCMAES();
  virtual ~IPOPCMAES();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);

  bool restart();
  virtual void optimize();
  virtual bool step();

  Eigen::VectorXd getNext();

  void setError(double fitness);

  bool terminated();
  virtual Eigen::VectorXd result();
  virtual std::string name();

  void setSigma0(double sigma0);
};

}

#endif
