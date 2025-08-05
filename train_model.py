import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv('data/sample_data.csv')

# Encode 'Shift'
le = LabelEncoder()
df['Shift'] = le.fit_transform(df['Shift'])  # Day=0, Night=1

# Features & target
X = df[['Exposure_mSv', 'Hours_Worked', 'Shift']]
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model, 'model/rf_model.pkl')
print("✅ Model trained and saved.")