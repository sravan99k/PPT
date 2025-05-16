import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your data
df = pd.read_csv("Student_Survey_Responses_300.csv", encoding="ISO-8859-1")
# Strip whitespace from column names and print them for debugging
print("Columns in the dataset:")
df.columns = df.columns.str.strip()
print(df.columns.tolist())

# Define risk score mapping
risk_score_map = {
    "Always": 1.0, "Often": 0.75, "Sometimes": 0.5,
    "Rarely": 0.25, "Never": 0.0, "Not Sure": 0.5,
    "Skip": 0.5, "Yes": 1.0, "No": 0.0
}

# Define survey questions grouped by disorder category
disorder_categories = {
    "Stress": [
        "I feel overwhelmed by my emotions",
        "I often feel anxious",
        "I often feel lonely or tearful"
    ],
    "Depression": [
        "I have felt hopeless or helpless recently",
        "I feel like life is not worth living",
        "I have thoughts of hurting myself"
    ],
    "Eating Disorder": [
        "I worry excessively about gaining weight",
        "I feel pressure to look a certain way because of social media or peers",
        "I restrict food intake to control my weight",
        "I skip meals intentionally",
        "I eat even when I'm not hungry due to stress or emotions",
        "I feel guilty after eating",
        "I avoid eating in front of others",
        "Do you think your eating habits affect your emotional or physical well-being?"
    ],
    "Behavioral Issues": [
        "I get into fights with my classmates or friends",
        "I skip school or classes without a good reason",
        "I tend to lie or hide the truth to avoid trouble",
        "I have trouble following rules or instructions",
        "I find it difficult to share my feelings with others"
    ]
}

# Apply scoring (map answers to numerical values)
for disorder, questions in disorder_categories.items():
    df[disorder + " %"] = df[questions].apply(lambda x: x.map(lambda x: risk_score_map.get(str(x).strip(), 0.5))).mean(axis=1) * 100

# Compute overall mental health risk
df["Mental Health Risk %"] = df[[col for col in df.columns if col.endswith(" %")]].mean(axis=1)

# Prepare data for model
X = df[[col for col in df.columns if col.endswith(" %") and col != "Mental Health Risk %"]]
y = df["Mental Health Risk %"]

# Binarize the target: classify as Low, Moderate, High
def classify_risk(percent):
    if percent < 30:
        return "Low"
    elif 30 <= percent < 60:
        return "Moderate"
    else:
        return "High"

y_class = y.apply(classify_risk)

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_class)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train, y_train)

# Predict class labels
y_pred = clf_model.predict(X_test)

# Evaluate model
print("📊 Classification Report:")
# Ensure the classes are present in both y_true and y_pred
class_labels = [0, 1, 2]  # Low=0, Moderate=1, High=2
print(classification_report(y_test, y_pred, target_names=le.classes_, labels=class_labels))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save final results with predicted classifications and guidance
df["Predicted Risk %"] = clf_model.predict(X)  # Predict for all data points
df["Predicted Mental Health Status"] = le.inverse_transform(df["Predicted Risk %"])

# Guidance generation
def personalized_guidance(row):
    guidance = []

    if row["Stress %"] > 60:
        guidance.append(" High stress detected. Try relaxation techniques like deep breathing, yoga, or regular physical activity. Maintain a consistent sleep schedule and talk to a school counselor or trusted adult.")

    if row["Depression %"] > 60:
        guidance.append("Signs of possible depression. You should speak with a mental health professional. Maintain a support network, reduce screen time, and engage in meaningful activities that bring joy.")

    if row["Eating Disorder %"] > 60:
        guidance.append("Eating behavior concerns found. Practice balanced eating, avoid crash dieting, and limit social media influence. Consider seeing a nutritionist or counselor specializing in body image.")

    if row["Behavioral Issues %"] > 60:
        guidance.append("Behavioral difficulties noticed. Practice self-awareness techniques like journaling, avoid impulsive decisions, and seek mentorship. Conflict resolution workshops may help.")

    if not guidance:
        return "✅ You are doing well across all areas. Keep practicing positive mental health habits like regular exercise, talking about your feelings, balanced nutrition, and mindfulness."

    return " ".join(guidance)

df["Suggested Guidance"] = df.apply(personalized_guidance, axis=1)

# Save final results including class/grade if available
output_cols = list(df.columns)
# Find class/grade column, case-insensitive
possible_class_cols = [col for col in df.columns if col.lower() in ["class", "grade"]]
class_col = possible_class_cols[0] if possible_class_cols else None
if class_col:
    output_cols.insert(1, output_cols.pop(output_cols.index(class_col)))
df.to_csv("Final_Mental_Health_Report.csv", index=False, columns=output_cols)

# Print average risk by class/grade if present
if class_col:
    avg_risk_by_class = df.groupby(class_col)["Mental Health Risk %"].mean()
    print(f"\nAverage Mental Health Risk % by {class_col}:")
    print(avg_risk_by_class)
else:
    print("No class/grade column found in the data.")
