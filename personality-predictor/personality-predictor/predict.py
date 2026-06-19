import joblib

model = joblib.load("personality_model.pkl")

social_hours = float(input("Hours spent socially per week: "))
friends_count = int(input("Number of close friends: "))
party_frequency = int(input("Party frequency per month: "))

prediction = model.predict([[social_hours, friends_count, party_frequency]])

print("Predicted personality of sample:", prediction[0])
