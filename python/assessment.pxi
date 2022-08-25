def sse(learner, dataset):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  return cbindings.sse(deref(net), deref(ds))

def mse(learner, dataset):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  return cbindings.mse(deref(net), deref(ds))

def rmse(learner, dataset):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  return cbindings.rmse(deref(net), deref(ds))

def accuracy(learner, dataset):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  return cbindings.accuracy(deref(net), deref(ds))

def confusion_matrix(learner, dataset):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  cdef cbindings.MatrixXi conf_mat = cbindings.confusionMatrix(deref(net), deref(ds))
  return __matrix_eigen_to_numpy_int__(&conf_mat)

def classification_hits(learner, dataset):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  return cbindings.classificationHits(deref(net), deref(ds))

def cross_validation(folds, learner, dataset, optimizer):
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<DataSet?>dataset).storage
  cdef cbindings.Optimizer *opt = (<Optimizer?>optimizer).thisptr
  return cbindings.crossValidation(folds, deref(net), deref(ds), deref(opt))