import logging
import timeit 
import pytest
import numpy as np 
import tensorflow as tf
from sklearn import datasets
from pls import PLSRegression
from sklearn.cross_decomposition import PLSRegression

LOGGER = logging.getLogger(__name__)

# Setting up
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target
X_10 = np.copy(X)
Y_10 = np.copy(Y)
for i in range(10):
    X_10 = np.concatenate((X_10, X_10), axis=1)
    Y_10 = np.concatenate((Y_10, Y_10), axis=0)
X_20 = np.copy(X_10)
Y_20 = np.copy(Y_10)
for i in range(10):
    X_20 = np.concatenate((X_20, X_20), axis=1)
    Y_20 = np.concatenate((Y_20, Y_20), axis=0)
Xt = tf.convert_to_tensor(X,dtype=tf.float64)
Xt_10 = tf.convert_to_tensor(X_10,dtype=tf.float64)
Xt_20 = tf.convert_to_tensor(X_20,dtype=tf.float64)
Yt = tf.convert_to_tensor(Y,dtype=tf.float64) 
Yt_10 = tf.convert_to_tensor(Y_10,dtype=tf.float64) 
Yt_20 = tf.convert_to_tensor(Y_20,dtype=tf.float64) 
    
def test_sklean_pls_speed_X_1():
    LOGGER.info('Performance test sklearn standard iris')
    def pls_speed(): 
        plsSKLEARN = PLSRegression()
        plsSKLEARN.fit(X,Y)
    LOGGER.info("Sklearn pls run time :{} s".format(timeit.timeit(pls_speed, 
                    number = 10)))
    assert True

def test_sklean_pls_speed_X_10():
    LOGGER.info('Performance test sklearn standard iris')
    def pls_speed(): 
        plsSKLEARN = PLSRegression()
        plsSKLEARN.fit(X_10,Y)
    LOGGER.info("Sklearn pls run time :{} s".format(timeit.timeit(pls_speed, 
                    number = 10)))
    assert True
    
#def test_sklean_pls_speed_X_20():
#    LOGGER.info('Performance test sklearn standard iris')
#    def pls_speed(): 
#        plsSKLEARN = PLSRegression()
#        plsSKLEARN.fit(X_20,Y)
#    LOGGER.info("Sklearn pls run time :{} s".format(timeit.timeit(pls_speed, 
#                    number = 10)))
#    assert True

def test_tensorflow_pls_speed_X_1():
    LOGGER.info('Performance test sklearn standard iris')
    def pls_speed(): 
        plsTENSORFLOW = PLSRegression()
        plsTENSORFLOW = plsTENSORFLOW.fit(Xt,Yt)
    LOGGER.info("Tensorflow pls run time :{} s".format(timeit.timeit(pls_speed, 
                    number = 10)))
    assert True
    
def test_tensorflow_pls_speed_X_10():
    LOGGER.info('Performance test sklearn standard iris')
    def pls_speed(): 
        plsTENSORFLOW = PLSRegression()
        plsTENSORFLOW = plsTENSORFLOW.fit(Xt,Yt)
    LOGGER.info("Tensorflow pls run time :{} s".format(timeit.timeit(pls_speed, 
                    number = 10)))
    assert True
    
#def test_tensorflow_pls_speed_X_20():
#    LOGGER.info('Performance test sklearn standard iris')
#    def pls_speed(): 
#        plsTENSORFLOW = PLSRegression()
#        plsTENSORFLOW = plsTENSORFLOW.fit(Xt,Yt)
#    LOGGER.info("Sklearn pls run time :{} s".format(timeit.timeit(pls_speed, 
#                    number = 10)))
#    assert True
    
    
def test_sklearn_pls_speed_step():
    LOGGER.info('Performance test sklearn standard iris')
    X_x = np.copy(X)
    Y_x = np.copy(Y)
    run_time = []
    for i in range(10):
        X_x = np.concatenate((X_x, X_x), axis=1)
        Y_x = np.concatenate((Y_x, Y_x), axis=0)
        def pls_speed(): 
            plsSKLEARN = PLSRegression()
            plsSKLEARN.fit(X_x,Y_x)
        run_time.append((timeit.timeit(pls_speed, number = 10)))
    LOGGER.info("Sklearn pls run times :{} s".format(run_time))
    assert True
    
def test_tensorflow_pls_speed_step():
    LOGGER.info('Performance test sklearn standard iris')
    X_x = np.copy(X)
    Y_x = np.copy(Y)
    run_time = []
    for i in range(10):
        X_x = tf.convert_to_tensor(np.concatenate((X_x, X_x), axis=1),dtype=tf.float64)
        Y_x = tf.convert_to_tensor(np.concatenate((Y_x, Y_x), axis=0),dtype=tf.float64)
        Xt_x = tf.convert_to_tensor(np.copy(X_x),dtype=tf.float64)
        Yt_x = tf.convert_to_tensor(np.copy(Y_x),dtype=tf.float64)
        def pls_speed(): 
            plsTENSORFLOW = PLSRegression()
            plsTENSORFLOW = plsTENSORFLOW.fit(Xt_x,Yt_x)
        run_time.append((timeit.timeit(pls_speed, number = 10)))
    LOGGER.info("Sklearn pls run times :{} s".format(run_time))
    assert True