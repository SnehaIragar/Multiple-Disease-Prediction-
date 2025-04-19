import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with the actual file)
# The dataset should have features (symptoms, test results, etc.)
# and a target variable indicating the disease(s).
# If a patient can have multiple diseases, this becomes a multi-label classification problem.
# This example assumes a single target variable for simplicity.
data = pd.read_csv('your_data.csv')

# Separate features (X) and target (y)
# Adjust column names based on your dataset
X = data.drop('disease', axis=1)
y = data['disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a machine learning model
# Here, we use Random Forest as an example. You can try other classifiers.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Disease')
plt.ylabel('Actual Disease')
plt.title('Confusion Matrix')
plt.show()

# --- Function for making predictions on new data ---
def predict_disease(input_data):
    """
    Predicts the disease based on the input features.

    Args:
        input_data (list or numpy array): A list or array of feature values
                                         in the same order as the training data.

    Returns:
        str: The predicted disease.
    """
    # Ensure the input data is in the correct format (e.g., a 2D array)
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)[0]
    return prediction

# Example of using the prediction function
new_patient_data = [/* Enter feature values here in the correct order */]
predicted_disease = predict_disease(new_patient_data)
print(f"\nPredicted Disease for the new patient: {predicted_disease}")
