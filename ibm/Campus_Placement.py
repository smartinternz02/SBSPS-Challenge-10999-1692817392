import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'data.csv' with your dataset's filename)
data = pd.read_csv('data/placement_data.csv')

# Data preprocessing
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['ssc_b'] = label_encoder.fit_transform(data['ssc_b'])
data['hsc_b'] = label_encoder.fit_transform(data['hsc_b'])
data['hsc_s'] = label_encoder.fit_transform(data['hsc_s'])
data['degree_t'] = label_encoder.fit_transform(data['degree_t'])
data['workex'] = label_encoder.fit_transform(data['workex'])
data['specialisation'] = label_encoder.fit_transform(data['specialisation'])
data['status'] = label_encoder.fit_transform(data['status'])

# Split data into features (X) and target (y)
X = data.drop(['status', 'sl_no', 'salary'], axis=1)
y = data['status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model (Logistic Regression in this case)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)