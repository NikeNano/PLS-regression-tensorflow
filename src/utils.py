def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
        """Perform is_fitted validation for estimator.
        Checks if the estimator is fitted by verifying the presence of
        "all_or_any" of the passed attributes and raises a NotFittedError with the
        given message.
        Parameters
        ----------
        estimator : estimator instance.
            estimator instance for which the check is performed.
        attributes : attribute name(s) given as string or a list/tuple of strings
            Eg.:
                ``["coef_", "estimator_", ...], "coef_"``
        msg : string
            The default error message is, "This %(name)s instance is not fitted
            yet. Call 'fit' with appropriate arguments before using this method."
            For custom messages if "%(name)s" is present in the message string,
            it is substituted for the estimator name.
            Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
        all_or_any : callable, {all, any}, default all
            Specify whether all or any of the given attributes must exist.
        Returns
        -------
        None
        Raises
        ------
        NotFittedError
            If the attributes are not found.
        """
        if msg is None:
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

        if not hasattr(estimator, 'fit'):
            raise TypeError("%s is not an estimator instance." % (estimator))

        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]

        if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(msg % {'name': type(estimator).__name__})  