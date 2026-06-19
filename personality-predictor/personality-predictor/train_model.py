import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("personality_data.csv")

X = data[["social_hours", "friends_count", "party_frequency"]]
y = data["personality"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "personality_model.pkl")

print("Model trained successfully.")
