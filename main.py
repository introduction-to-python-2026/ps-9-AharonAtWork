# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd

# 'file_name.extension' should be the path to your file
df = pd.read_csv('/content/parkinsons.csv')
input_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
output_feature = 'status'

X = df[input_features]
y = df[output_feature]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5) # Using K-Nearest Neighbors with 5 neighbors
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

if accuracy >= 0.8:
    print("✅ Model accuracy is at least 0.8!")
else:
    print("⚠️ Model accuracy is below 0.8. Consider re-evaluating the model or features.")
    
