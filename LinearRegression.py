import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Titanic dataset
file_path = 'Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Preprocessing: Select features and target variable
# Predict 'Fare' based on 'Age' and 'Pclass'
df = df[['Fare', 'Age', 'Pclass']].dropna()

# Define features and target
X = df[['Age', 'Pclass']]
y = df['Fare']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot regression line for Age vs Fare (simple linear regression for visualization)
plt.scatter(X_test['Age'], y_test, color='blue', label='Actual Fare')
plt.scatter(X_test['Age'], y_pred, color='red', label='Predicted Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Linear Regression: Age vs Fare')
plt.legend()
plt.show()

# Print coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)
print("Coefficients (Age, Pclass):", coefficients)
print("Intercept:", intercept)

