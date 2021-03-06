import tensorflow as tf
import tensorflow_probability as tfp

# from utils import check_is_fitted
from abc import ABCMeta, abstractmethod

#@tf.function
def _nipals_tensorflow(
    X: tf.Tensor, Y: tf.Tensor, max_iter=500, tol=1e-06, norm_y_weights: bool = False
):
    """
    The inner look of the nipals algorhtim 

    input:
        X               - tf.Tensor,shape=(row,columns), the input features
        Y               - tf.Tensor, shape=(row,columsn), the dependent variable
        max_itter       - int, the max number of itterations
        tol             - float, the minimum tolerance needed to stop
        norm_y_weights  - boolean, normalising the y_weights during each itteration
    
    output:
        x_weights       -tf.Tensor,shape(1,nbr of columns in X) the output weights
        y_weights       -tf.Tensor,shape(1,nbr of columns in X) the output weights
        ite             -int, the number of itterations
    """

    norm_y_weights = tf.convert_to_tensor(norm_y_weights)
    y_score = Y
    x_weights_old = tf.constant(0, shape=[X.shape[1], 1], dtype=X.dtype)
    ite = 1
    X_pinv = Y_pinv = None
    eps = tf.keras.backend.epsilon()
    # Inner loop of the Wold algo.
    x_weights = tf.constant(0, shape=[X.shape[1], 1], dtype=X.dtype)
    y_weights = tf.constant(0, shape=[1, 1], dtype=X.dtype)
    while tf.constant(True):
        # 1. Regress into X
        x_weights = tf.matmul(tf.transpose(X), y_score) / (
            tf.matmul(tf.transpose(y_score), y_score)
        )
        # 1.2 Normalize u
        x_weights = x_weights / tf.math.sqrt(tf.matmul(tf.transpose(x_weights), x_weights))

        # 1.3 Update x_score: the X latent scores
        x_score = tf.matmul(X, x_weights) / tf.matmul(tf.transpose(x_weights), x_weights)
        # 2. Regress into Y
        y_weights = tf.matmul(tf.transpose(Y), x_score) / (
            tf.matmul(tf.transpose(x_score), x_score)
        )
        # 2.1 Normalize y_weights
        if norm_y_weights is not None:
            y_weights /= y_weights / tf.math.sqrt(tf.matmul(tf.transpose(y_weights), y_weights))
        # 2.2 Update y_score: the Y latent scores
        y_score = tf.matmul(Y, y_weights) / tf.matmul(tf.transpose(y_weights), y_weights)
        x_weights_diff = tf.math.subtract(x_weights, x_weights_old)
        if tf.matmul(tf.transpose(x_weights_diff), x_weights_diff)[0][0] < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            warnings.warn("Maximum number of iterations reached", ConvergenceWarning)
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite


@tf.function
def _center_scale_xy(X: tf.Tensor, Y: tf.Tensor, scale=True):
    """ Center X, Y and scale if the scale parameter==True
    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = tf.reduce_mean(X, axis=0)
    X -= x_mean
    y_mean = tf.reduce_mean(Y, axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = tf.math.reduce_std(X, axis=0)
        x_std = x_std = tf.where(tf.equal(x_std, 0), tf.ones_like(x_std), x_std)
        X /= x_std
        y_std = tf.math.reduce_std(Y, axis=0)
        y_std = x_std = tf.where(tf.equal(y_std, 0), tf.ones_like(y_std), y_std)
        Y /= y_std
    else:
        x_std = tf.ones(X.shape[1])
        y_std = tf.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


class _PLS:
    @abstractmethod
    def __init__(
        self,
        n_components=2,
        scale=True,
        deflation_mode="regression",
        mode="A",
        algorithm="nipals",
        norm_y_weights=False,
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy


    def fit(self, X: tf.Tensor, Y: tf.Tensor):  # does this work?
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        self.x_scores_ = []
        self.y_scores_ = []
        self.x_weights_ = []
        self.y_weights_ = []
        self.x_loadings_ = []
        self.y_loadings_ = []

        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = _center_scale_xy(
            X, Y, self.scale
        )
        # Residuals (deflated) matrices
        Xk = X
        Yk = Y
        # Results matrices
        # self.x_scores_ = tf.zeros(shape=[n, self.n_components])
        # self.y_scores_ = tf.zeros(shape=[n, self.n_components])
        # self.x_weights_ = tf.zeros(shape=[p, self.n_components])
        # self.y_weights_ = tf.zeros(shape=[q, self.n_components])
        # self.x_loadings_ = tf.zeros(shape=[p, self.n_components])
        # self.y_loadings_ = tf.zeros(shape=[q, self.n_components])
        self.n_iter_ = []

        # NIPALS algo: outer loop, over components
        Y_eps = tf.keras.backend.epsilon()
        for k in range(self.n_components):
            #             if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):  #TO IMPLEMENT TENSORFLOW YET
            #                 # Yk constant
            #                 warnings.warn('Y residual constant at iteration %s' % k)
            #                 break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                # Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                # Yk[:, Yk_mask] = 0.0

                x_weights, y_weights, n_iter_ = _nipals_tensorflow(
                    X=Xk,
                    Y=Yk,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    norm_y_weights=self.norm_y_weights,
                )
                self.n_iter_.append(n_iter_)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            # x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            # y_weights = y_weights.T REPLACED DOWN HERE
            y_weights = tf.transpose(y_weights)
            # compute scores
            x_scores = tf.matmul(Xk, x_weights)
            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = tf.matmul(tf.transpose(y_weights), y_weights)
            y_scores = tf.matmul(Yk, y_weights) / y_ss
            # test for null variance
            if tf.matmul(tf.transpose(x_scores), x_scores) < Y_eps:
                break
            # 2) Deflation (in place)
            # ----------------------
            # Possible memory footprint reduction may done here: in order to
            # avoid the allocation of a data chunk for the rank-one
            # approximations matrix which is then subtracted to Xk, we suggest
            # to perform a column-wise deflation.
            #
            # - regress Xk's on x_score
            # x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            x_loadings = tf.matmul(tf.transpose(Xk), x_scores) / tf.matmul(
                tf.transpose(x_scores), x_scores
            )
            # - subtract rank-one approximations to obtain remainder matrix
            # Xk -= np.dot(x_scores, x_loadings.T)
            Xk -= tf.matmul(x_scores, tf.transpose(x_loadings))
            y_loadings = []
            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = np.dot(Yk.T, y_scores) / np.dot(y_scores.T, y_scores)
                Yk -= np.dot(y_scores, y_loadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                #                 y_loadings = (np.dot(Yk.T, x_scores)
                #                               / np.dot(x_scores.T, x_scores))
                #                 Yk -= np.dot(x_scores, y_loadings.T)
                y_loadings = tf.matmul(tf.transpose(Yk), x_scores) / tf.matmul(
                    tf.transpose(x_scores), x_scores
                )
                Yk -= tf.matmul(x_scores, tf.transpose(y_loadings))

            # 3) Store weights, scores and loadings # Notation:
            #             self.x_scores_[:, k] = x_scores.ravel()  # T
            #             self.y_scores_[:, k] = y_scores.ravel()  # U
            #             self.x_weights_[:, k] = x_weights.ravel()  # W
            #             self.y_weights_[:, k] = y_weights.ravel()  # C
            #             self.x_loadings_[:, k] = x_loadings.ravel()  # P
            #             self.y_loadings_[:, k] = y_loadings.ravel()  # Q
            #             x_scores = tf.dtypes.cast(x_scores,tf.float32)
            #             y_scores = tf.dtypes.cast(y_scores,tf.float32)
            #             x_weights = tf.dtypes.cast(x_weights,tf.float32)
            #             y_weights = tf.dtypes.cast(y_weights,tf.float32)
            #             x_loadings = tf.dtypes.cast(x_loadings,tf.float32)
            #             y_loadings = tf.dtypes.cast(y_loadings,tf.float32)
            self.x_scores_.append(x_scores)
            self.y_scores_.append(y_scores)
            self.x_weights_.append(x_weights)
            self.y_weights_.append(y_weights)
            self.x_loadings_.append(x_loadings)
            self.y_loadings_.append(y_loadings)
        # Such that: X = TP' + Err and Y = UQ' + Err
        self.x_scores_ = tf.stack(self.x_scores_, axis=1)
        self.x_scores_ = tf.reshape(self.x_scores_, shape=self.x_scores_.shape[:2])
        self.y_scores_ = tf.stack(self.y_scores_, axis=1)
        self.y_scores_ = tf.reshape(self.y_scores_, shape=self.y_scores_.shape[:2])
        self.x_weights_ = tf.stack(self.x_weights_, axis=1)
        self.x_weights_ = tf.reshape(self.x_weights_, shape=self.x_weights_.shape[:2])
        self.y_weights_ = tf.stack(self.y_weights_, axis=1)
        self.y_weights_ = tf.reshape(self.y_weights_, shape=self.y_weights_.shape[:2])
        self.x_loadings_ = tf.stack(self.x_loadings_, axis=1)
        self.x_loadings_ = tf.reshape(self.x_loadings_, shape=self.x_loadings_.shape[:2])
        self.y_loadings_ = tf.stack(self.y_loadings_, axis=1)
        self.y_loadings_ = tf.reshape(self.y_loadings_, shape=self.y_loadings_.shape[:2])

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = tf.matmul(
            self.x_weights_,
            tfp.math.pinv(tf.matmul(tf.transpose(self.x_loadings_), self.x_weights_)),
        )

        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False),
            )
        else:
            self.y_rotations_ = tf.ones(1, dtype=X.dtype)

        if True or self.deflation_mode == "regression":
            # FIXME what's with the if?
            # Estimate regression coefficient
            # Regress Y on T
            # Y = TQ' + Err,
            # Then express in function of X
            # Y = X W(P'W)^-1Q' + Err = XB + Err
            # => B = W*Q' (p x q)
            self.coef_ = tf.matmul(self.x_rotations_, tf.transpose(self.y_loadings_))

            self.coef_ = self.coef_ * self.y_std_
        return self

    @tf.function
    def transform(self, X: tf.Tensor, Y: tf.Tensor = None, copy=True):
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        Y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """

        # check_is_fitted(self, 'x_mean_')
        # [TODO] fix this later, needs more work ...
        # X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = tf.matmul(
            tf.dtypes.cast(X, tf.float32), tf.dtypes.cast(self.x_rotations_, tf.float32)
        )
        if Y is not None:
            # [TODO] fix this later, needs more work ...
            # Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.shape[0] == 1:
                Y = tf.reshape(Y, shape=(-1, 1))
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = tf.tensordot(Y, self.y_rotations_, axes=0)
            return x_scores, y_scores

        return x_scores

    @tf.function
    def predict(self, X: tf.Tensor, copy=True) -> tf.Tensor:
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        # check_is_fitted(self)
        # X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Ypred = np.dot(X, self.coef_)
        Ypred = tf.matmul(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X: tf.Tensor, y: tf.Tensor = None):
        """Learn and apply the dimension reduction on the train data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {"poor_score": True}


class PLSRegression(_PLS):
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="regression",
            mode="A",
            norm_y_weights=False,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )

