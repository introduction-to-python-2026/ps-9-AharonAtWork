import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('/content/parkinsons.csv')

# 2. Select features
input_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
output_feature = 'status'

X = df[input_features]
y = df[output_feature]

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Choose and train a model (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 6. Test the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

if accuracy >= 0.8:
    print("✅ Model accuracy is at least 0.8!")
else:
    print("⚠️ Model accuracy is below 0.8. Consider re-evaluating the model or features.")

# 7. Save the model
joblib.dump(model, 'my_model.joblib')
print("Model saved as my_model.joblib")
