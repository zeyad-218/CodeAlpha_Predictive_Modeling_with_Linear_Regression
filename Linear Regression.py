import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read data
file_path = "/Users/zeyadosama/Desktop/CodeAlpha/Task2/insurance.csv"
insurance = pd.read_csv(file_path)
print(insurance.info())

# One-hot encoding for categorical variables
insurance = pd.get_dummies(insurance, drop_first=True)

# Data visualization
sns.pairplot(insurance)
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(insurance.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Split the data into training set and testing set
np.random.seed(144)
Train, Test = train_test_split(insurance, test_size=0.3, random_state=144)

# Accuracy of Baseline Model on testing set:
SST = np.sum((Test['charges'] - np.mean(insurance['charges'])) ** 2)
print("SST:", SST)

# Build Linear Regression Model using all other variables as independent variables:
Linear_Regression_Model = LinearRegression()
X_train = Train.drop('charges', axis=1)
y_train = Train['charges']
X_test = Test.drop('charges', axis=1)
y_test = Test['charges']

Linear_Regression_Model.fit(X_train, y_train)
Prediction = Linear_Regression_Model.predict(X_test)

# Calculate Out-of-Sample SSE and R2:
SSE = np.sum((y_test - Prediction) ** 2)
R2 = 1 - SSE/SST
print("Linear Regression R2:", R2)

# Data visualization: Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, Prediction, alpha=0.6, color='b')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()

# Residuals
residuals = y_test - Prediction
plt.figure(figsize=(10, 6))
plt.scatter(Prediction, residuals, alpha=0.6, color='b')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Charges')
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Model summary
print("Mean Squared Error:", mean_squared_error(y_test, Prediction))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, Prediction)))
print("R2 Score:", r2_score(y_test, Prediction))