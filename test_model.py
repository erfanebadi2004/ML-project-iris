import joblib

with open("random_farest.joblib", "rb") as model:
    rfc = joblib.load(model)

test_data = [[2.14, 6.4, 1.7, 3.75]]

result = rfc.predict(test_data)

print(f"prediction results : {result}")

print("The prediction was successfully made.")
