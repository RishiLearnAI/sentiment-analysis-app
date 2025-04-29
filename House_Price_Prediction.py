
# ===============================
# Mini Project: House Price Prediction
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a simple dataset
data = {
    'area': [2600, 3000, 3200, 3600, 4000],
    'bedrooms': [3, 4, 3, 3, 5],
    'age': [20, 15, 18, 30, 8],
    'price': [550000, 565000, 610000, 595000, 760000]
}

df = pd.DataFrame(data)

# Step 2: Visualize the data
plt.scatter(df['area'], df['price'], color='blue')
plt.title("House Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()

# Step 3: Prepare the data
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: Predict a new house
new_house = [[3500, 4, 10]]  # area, bedrooms, age
predicted_price = model.predict(new_house)
print(f"Predicted price for house {new_house}: ${predicted_price[0]:,.2f}")
