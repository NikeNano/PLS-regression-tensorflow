import tensorflow as tf

@tf.function
def _nipals_tensorflow(X,Y, max_iter=500, tol=1e-06):
    y_score = Y
    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = tf.keras.backend.epsilon()
    # Inner loop of the Wold algo.
    while True:
        #1. Regress into X
        x_weights = tf.matmul(tf.transpose(X),y_score)/(tf.matmul(tf.transpose(y_score),y_score))
        # 1.2 Normalize u
        x_weights = x_weights/tf.math.sqrt(tf.matmul(tf.transpose(x_weights),x_weights))
        
        # 1.3 Update x_score: the X latent scores
        x_score = tf.matmul(X,x_weights)/tf.matmul(tf.transpose(x_weights),x_weights)
        #2. Regress into Y
        y_weights = tf.matmul(tf.transpose(Y),x_score)/(tf.matmul(tf.transpose(x_score),x_score))
        # 2.1 Normalize y_weights
        y_weights /= y_weights/tf.math.sqrt(tf.matmul(tf.transpose(y_weights),y_weights))
        # 2.2 Update y_score: the Y latent scores
        y_score = tf.matmul(Y,y_weights)/tf.matmul(tf.transpose(y_weights),y_weights)
        x_weights_diff = tf.math.subtract(x_weights,x_weights_old)
        if tf.matmul(tf.transpose(x_weights_diff),x_weights_diff) < tol or Y.shape[1] ==1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite

