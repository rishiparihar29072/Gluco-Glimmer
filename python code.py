import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# To ensure reproducibility
np.random.seed(42)

# Generate synthetic sensor data
num_samples = 1000
skin_temp = np.random.normal(loc=35, scale=0.5, size=num_samples)
sweat_composition = np.random.normal(loc=1, scale=0.2, size=num_samples)
heart_rate = np.random.normal(loc=70, scale=5, size=num_samples)
glucose_level = 4 * skin_temp + 2 * sweat_composition + 0.5 * heart_rate + np.random.normal(loc=0, scale=2, size=num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'skin_temp': skin_temp,
    'sweat_composition': sweat_composition,
    'heart_rate': heart_rate,
    'glucose_level': glucose_level
})

# Split data into training and testing sets
X = data[['skin_temp', 'sweat_composition', 'heart_rate']]
y = data['glucose_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot true vs predicted glucose levels
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Glucose Level')
plt.ylabel('Predicted Glucose Level')
plt.title('True vs Predicted Glucose Level')
plt.show()
