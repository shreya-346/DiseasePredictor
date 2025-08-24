import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np
from rapidfuzz import process, fuzz  # for fuzzy matching

# -----------------------------
#  Load dataset
# -----------------------------
data = pd.read_csv("dataset.csv")
data = data.drop(columns=["Unnamed: 133"])

X = data.drop(columns=["prognosis"])
y = data["prognosis"]

# -----------------------------
#  Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model and encoder
# -----------------------------
with open("disease_model.pkl", "wb") as file:
    pickle.dump(model, file)
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)

# -----------------------------
#  Fuzzy symptom matching
# -----------------------------
all_symptoms = list(X.columns)

def match_symptoms(user_input, all_symptoms, threshold=80):
    """
    Matches user input symptoms to dataset symptoms using fuzzy matching.
    Returns a list of matched symptoms.
    """
    user_symptoms = [sym.strip().lower() for sym in user_input.split(",")]
    matched = []
    for symptom in user_symptoms:
        match, score, _ = process.extractOne(symptom, all_symptoms, scorer=fuzz.WRatio)
        if score >= threshold:
            matched.append(match)
    return matched

# -----------------------------
#  Predict top 3 diseases
# -----------------------------
def predict_top3(user_input):
    matched_symptoms = match_symptoms(user_input, all_symptoms)
    
    if not matched_symptoms:
        print("No matching symptoms found. Try again with different words.")
        return
    
    # Create input vector
    input_vector = [1 if symptom in matched_symptoms else 0 for symptom in all_symptoms]
    input_vector = [input_vector]
    
    # Predict probabilities
    probabilities = model.predict_proba(input_vector)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    top3_diseases = le.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]
    
    print("\nTop 3 most likely diseases:")
    for disease, prob in zip(top3_diseases, top3_probs):
        print(f"{disease}: {prob*100:.2f}%")

# -----------------------------
#  Take input and predict
# -----------------------------
user_input = input("Enter your symptoms separated by commas: ")
predict_top3(user_input)
