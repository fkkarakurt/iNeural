#include <iNeural/layers/Input.h>
#include <iNeural/util/Random.h>

namespace iNeural
{

    Input::Input(int dim1, int dim2, int dim3)
        : J(dim1 * dim2 * dim3), dim1(dim1), dim2(dim2), dim3(dim3), x(0)
    {
    }

    OutputInfo Input::initialize(std::vector<double *> &parameterPointers,
                                 std::vector<double *> &parameterDerivativePointers)
    {
        OutputInfo info;
        info.dimensions.push_back(dim1);
        info.dimensions.push_back(dim2);
        info.dimensions.push_back(dim3);
        return info;
    }

    void Input::initializeParameters()
    {
        //
    }

    void Input::forwardPropagate(Eigen::MatrixXd *x, Eigen::MatrixXd *&y,
                                 bool dropout, double *error)
    {
        this->x = x;
        y = this->x;
    }

    void Input::backpropagate(Eigen::MatrixXd *ein, Eigen::MatrixXd *&eout,
                              bool backpropToPrevious)
    {
        //
    }

    Eigen::MatrixXd &Input::getOutput()
    {
        return *x;
    }

    Eigen::VectorXd Input::getParameters()
    {
        return Eigen::VectorXd();
    }

}