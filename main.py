# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd
df = pd.read_csv('parkinsons.csv')
# Define input features
input_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']

# Define output feature
output_feature = 'status'

print(f"Input Features: {input_features}")
print(f"Output Feature: {output_feature}")
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply scaling to the input features
df[input_features] = scaler.fit_transform(df[input_features])
from sklearn.model_selection import train_test_split

X = df[input_features]
y = df[output_feature]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model
model = KNeighborsClassifier(n_neighbors=5)

from sklearn.metrics import accuracy_score

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

if accuracy >= 0.8:
    print("Accuracy target of 0.8 met!")
else:
    print("Accuracy target of 0.8 not met. Consider re-evaluating features or model.")
