import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("student_data.csv")

print("Dataset Preview:")
print(data.head())

# Features and target
X = data[['G1', 'G2']]   # First and second period grades
y = data['G3']          # Final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization (G2 vs G3 for simple view)
plt.scatter(X_test['G2'], y_test, label="Actual")
plt.scatter(X_test['G2'], y_pred, label="Predicted")
plt.xlabel("G2 Grade")
plt.ylabel("G3 Final Grade")
plt.title("Linear Regression: Predicting Final Grade")
plt.legend()
plt.show()
