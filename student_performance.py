
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Calculate average student score
score_columns = ["math score", "reading score", "writing score"]
data["average score"] = data[score_columns].mean(axis=1)

# Select input features and target
X = data.drop(columns=score_columns + ["average score"])
y = data["average score"]

# Identify categorical and numerical features
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocess categorical features
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ("numerical", "passthrough", numerical_columns)
])

# Create the prediction model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict scores
predictions = model.predict(X_test)

# Evaluate model performance
print("\nSTUDENT PERFORMANCE PREDICTION RESULTS")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# Save predictions
results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": predictions.round(2)
})
results.to_csv("prediction_results.csv", index=False)
print("\nPredictions saved to prediction_results.csv")

# Plot actual versus predicted scores
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.7)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Student Performance Prediction")
plt.tight_layout()
plt.savefig("prediction_plot.png")
plt.show()
