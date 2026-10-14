# Student Performance Prediction Using Machine Learning

## 🚀 Live Demo

Try the application online: [**Student Performance Predictor**](https://student-performance-prediction-w2tgs8wnria6doit5xmerb.streamlit.app/)

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Random Forest Regressor

## ✨ Features

- Interactive student details form
- Predicts estimated average exam scores
- Displays model evaluation metrics
- Simple, user-friendly web interface

## Project Overview
This mini-project uses machine learning to predict students' average academic scores based on their demographic information and educational background.

## Dataset
The project uses the Students Performance dataset, which contains information about students' gender, parental education, lunch type, test preparation, and mathematics, reading, and writing scores.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Machine Learning Algorithm
-Random Forest Regressor

## Implementation
1. Load the student performance dataset.
2. Preprocess the data and select relevant features.
3. Split the dataset into training and testing sets.
4. Train the Random Forest Regressor model.
5. Evaluate the model using MAE, RMSE, and R² score.
6. Save the predictions and generate a graph comparing actual and predicted scores.

## Results
- **Mean Absolute Error (MAE):** 11.49
- **Root Mean Squared Error (RMSE):** 14.82
- **R² Score:** -0.025

The current model shows poor predictive performance, as indicated by the negative R² score. Further feature engineering and model improvement are needed.

## Project Files
- `student_performance.py` – Model implementation
- `StudentsPerformance.csv` – Dataset
- `prediction_results.csv` – Actual and predicted scores
- `prediction_plot.png` – Prediction visualization
- `student_performance_graph.png` – Saved graph
- `requirements.txt` – Required Python libraries

## How to Run
1. Install Python.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run `python student_performance.py`.

## Conclusion
This project demonstrates the basic machine learning workflow for student performance prediction, including data preparation, model training, evaluation, and visualization.
