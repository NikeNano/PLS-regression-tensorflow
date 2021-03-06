import tensorflow as tf
import numpy as np
from pls import _nipals_tensorflow
from sklearn import datasets
from sklearn.utils import check_array
from sklearn.cross_decomposition.pls_ import _nipals_twoblocks_inner_loop

def test_tensorflo_version():
    assert tf.__version__ =='2.0.0-beta1',"Wrong tf version need 2"

def test__nipals_tensorflow():
    iris = datasets.load_iris()
    X = check_array(iris.data, dtype=np.float64,ensure_min_samples=2)
    Y = check_array(iris.target, dtype=np.float64,ensure_2d=False)
    if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
    Xt = tf.convert_to_tensor(X)
    Yt = tf.convert_to_tensor(Y)
    Xt2 = tf.Variable(Xt)
    x_weights, y_weights, ite = _nipals_tensorflow(X=Xt,Y=Yt,max_iter=500, tol=1e-06,norm_y_weights=False)
    assert ite.numpy() == 1, "Should only need one itteration since one dependent variable"
    assert type(Xt)== type(x_weights)
    assert type(Yt)== type(y_weights)
    # check that the value of X,Y has changed!!!!!!
    assert x_weights.shape[0] == X.shape[1], "Wrong output shape of X"
    assert y_weights.shape[0] == Y.shape[1], "Wrong output shape of Y"
    assert y_weights.shape[0] == 1, "Wrong output shape of Y"
    assert y_weights.shape[1] == 1, "Wrong output shape of Y"
    assert Xt.shape == Xt2.shape
    assert Xt != Xt2


def test_compare__nipals_tensorflow_and_sklearn():
    iris = datasets.load_iris()
    X = check_array(iris.data, dtype=np.float64,ensure_min_samples=2)
    Y = check_array(iris.target, dtype=np.float64,ensure_2d=False)
    if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
    x_weights_sk,y_weights_sk,itter_sklearn = _nipals_twoblocks_inner_loop(X,Y)
    Xt = tf.convert_to_tensor(X)
    Yt = tf.convert_to_tensor(Y)
    x_weights_tf, y_weights_tf, itter_tensorflow = _nipals_tensorflow(X=Xt,Y=Yt,max_iter=500, tol=1e-06,norm_y_weights=False)
    assert np.testing.assert_allclose(x_weights_sk,x_weights_tf.numpy() ,rtol=1e-10, atol=0)==None, "Difference between sklearn and our imp"
    assert np.testing.assert_allclose(y_weights_sk,y_weights_tf.numpy() ,rtol=1e-10, atol=0)==None, "Difference between sklearn and our imp"
    assert itter_sklearn==itter_tensorflow.numpy(), "Equal number of itteration, should be 1"