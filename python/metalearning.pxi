cdef class Adaraise:
  cdef cbindings.Adaraise *thisptr

  def __cinit__(self):
    self.thisptr = new cbindings.Adaraise()

  def __dealloc__(self):
    del self.thisptr

  def get_weights(self):
    cdef cbindings.VectorXd w_eigen = self.thisptr.getWeights()
    return __vector_eigen_to_numpy__(&w_eigen)

  def add_learner(self, learner):
    self.thisptr.addLearner(deref((<Learner>learner).learner))

  def set_optimizer(self, optimizer):
    self.thisptr.setOptimizer(deref((<Optimizer>optimizer).thisptr))

  def train(self, data_set):
    self.thisptr.train(deref((<DataSet>data_set).storage))

  def predict(self, x_numpy):
    x_numpy = numpy.atleast_2d(x_numpy)
    cdef cbindings.MatrixXd* x_eigen = __matrix_numpy_to_eigen__(x_numpy)
    cdef cbindings.MatrixXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __matrix_eigen_to_numpy__(&y_eigen)


cdef class Bagging:
  cdef cbindings.Bagging *thisptr

  def __cinit__(self, bag_size):
    self.thisptr = new cbindings.Bagging(bag_size)

  def __dealloc__(self):
    del self.thisptr

  def add_learner(self, learner):
    self.thisptr.addLearner(deref((<Learner>learner).learner))

  def set_optimizer(self, optimizer):
    self.thisptr.setOptimizer(deref((<Optimizer>optimizer).thisptr))

  def train(self, data_set):
    self.thisptr.train(deref((<DataSet>data_set).storage))

  def predict(self, x_numpy):
    x_numpy = numpy.atleast_2d(x_numpy)
    cdef cbindings.MatrixXd* x_eigen = __matrix_numpy_to_eigen__(x_numpy)
    cdef cbindings.MatrixXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __matrix_eigen_to_numpy__(&y_eigen)