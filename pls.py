import tensorflow as tf

@tf.function
def _nipals_tensorflow(X,Y, max_iter=500, tol=1e-06,norm_y_weights=False):
    """
    The inner look of the nipals algorhtim 

    input:
        X               - tf.tensor,shape=(row,columns), the input features
        Y               - tf.tensor, shape=(row,columsn), the dependent variable
        max_itter       - int, the max number of itterations
        tol             - float, the minimum tolerance needed to stop
        norm_y_weights  - boolean, normalising the y_weights during each itteration
    
    output:
        x_weights       -tf.tensor,shape(1,nbr of columns in X) the output weights
        y_weights       -tf.tensor,shape(1,nbr of columns in X) the output weights
        ite             -int, the number of itterations
    """

    norm_y_weights = tf.convert_to_tensor(norm_y_weights)
    y_score = Y
    x_weights_old = tf.constant(0,shape=[X.shape[1],1],dtype='float64')
    ite = 1
    X_pinv = Y_pinv = None
    eps = tf.keras.backend.epsilon()
    # Inner loop of the Wold algo.
    x_weights = tf.constant(0,shape=[X.shape[1],1],dtype='float64')
    y_weights = tf.constant(0,shape=[1,1],dtype='float64')
    while tf.constant(True):
        #1. Regress into X
        x_weights = tf.matmul(tf.transpose(X),y_score)/(tf.matmul(tf.transpose(y_score),y_score))
        # 1.2 Normalize u
        x_weights = x_weights/tf.math.sqrt(tf.matmul(tf.transpose(x_weights),x_weights))
        
        # 1.3 Update x_score: the X latent scores
        x_score = tf.matmul(X,x_weights)/tf.matmul(tf.transpose(x_weights),x_weights)
        #2. Regress into Y
        y_weights = tf.matmul(tf.transpose(Y),x_score)/(tf.matmul(tf.transpose(x_score),x_score))
        # 2.1 Normalize y_weights
        if(norm_y_weights):
            y_weights /= y_weights/tf.math.sqrt(tf.matmul(tf.transpose(y_weights),y_weights))
        # 2.2 Update y_score: the Y latent scores
        y_score = tf.matmul(Y,y_weights)/tf.matmul(tf.transpose(y_weights),y_weights)
        x_weights_diff = tf.math.subtract(x_weights,x_weights_old)
        if tf.matmul(tf.transpose(x_weights_diff),x_weights_diff)[0][0] < tol:# or Y.shape[1] ==1:
            break
        if ite == 1:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
        break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite





