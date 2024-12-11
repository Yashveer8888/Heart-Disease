import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
data = pd.read_csv("heart_disease.csv")
X = data[['age','cp','thalach']]  # Features
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as 'heart_disease_model.pkl'")
