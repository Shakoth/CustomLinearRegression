CustomAnalyticalLinearRegression

This class implements a Linear Regression model using analytical methods for parameter estimation and statistical testing.


Usage

model = CustomAnalyticalLinearRegression(fit_intercept=True)

model.fit(X, y)

predictions = model.predict(X)


Methods

__init__(self, fit_intercept=True)

Constructor for the class. fit_intercept is a boolean that determines whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).


fit(self, X, y)

Fits the model according to the given training data. X is the training input samples of shape (n_samples, n_features). y is the target values of shape (n_samples,).


predict(self, X)

Predicts the output for the given samples. X is the input samples of shape (n_samples, n_features).


RSS(self, y_true, y_pred)

Returns the residual sum of squares (RSS) for the given true and predicted values.


residual_variance(self, y_true, y_pred, p)

Returns the residual variance for the given true and predicted values and the number of predictors p.


w_var(self, y_true, y_pred, X)

Returns the variance of the coefficients.


w_standard_errors(self, y_true, y_pred, X)

Returns the standard errors of the coefficients.


t_stats(self)

Returns the t-statistics for a hypothesis test on the coefficients. Raises a NotFittedError if the model is not fitted.


t_test(self)

Returns the p-values for the t-tests on the coefficients. Raises a NotFittedError if the model is not fitted.


_add_intercept(self, X)

Private method to add an intercept column to the input data if fit_intercept is True.


Attributes

W

The fitted linear coefficients.


n_samples

The number of samples used to fit the model.


errors

The standard errors of the coefficients.


t

The t-statistics for a hypothesis test on the coefficients.


pvalues

The p-values for the t-tests on the coefficients.


results

results

A DataFrame containing the coefficients, standard errors, t-statistics, and p-values for each of the model's predictors. This attribute is only populated after the fit method is called.


__repr__(self)

Returns a string that represents the model. This method is called when the print function is used on the model instance.


Exceptions

NotFittedError

This error is raised when you try to call predict, t_stats, or t_test before fitting the model.


Example

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate a random regression problem
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the model
model = CustomAnalyticalLinearRegression()
model.fit(X_train, y_train)

# Print the model results
print(model.results)

# Predict the test data
y_pred = model.predict(X_test)
This model is useful for situations where you need more statistical information about your regression model than what is provided by default in libraries like scikit-learn. It's particularly useful for educational purposes and in-depth statistical analysis.


