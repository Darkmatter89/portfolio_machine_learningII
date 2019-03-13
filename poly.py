import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#=============================================================
# =================  Polynomial Regression ===================
#=============================================================


# Training set
x_train = [[1290], [1350], [1470], [1600], [1710]] #Size of houses: m2
y_train = [[1182], [1172], [1264], [1493], [1571]] #Electric consumed: KW Hrs/Mnth

# Testing set
x_test = [[1840], [1980], [2230], [2400]] #Size of houses: m2
y_test = [[1711], [1804], [1840], [1956]] #Electricity consumed: KW Hrs/Mnth

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(3000, 250, 1500)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
X_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Electricity usage increased on area')
plt.xlabel('Size of houses: m2')
plt.ylabel('Electricity consumed: KW Hrs/Mnth')
plt.axis([1200, 1750, 1000, 1700])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()

# If you execute the code, you will see that the simple linear regression model is plotted with
# a solid line. The quadratic regression model is plotted with a dashed line and evidently
# the quadratic regression model fits the training data better.
